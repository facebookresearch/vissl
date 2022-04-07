# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from vissl.models import build_model
from vissl.models.model_helpers import get_no_ddp_model


class DINOHook(ClassyHook):
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop

    def __init__(self):
        """
        This hook corresponds to the DINO: the framework proposed in the xxx paper.

        Called before each forward to get teacher outputs and after every iteration to update
        the momentum teacher, optionally  updating the teacher temperature.

        At the beginning of training i.e. after 1st forward call, the encoder is constructed.
        """
        super().__init__()
        self.teacher_temp_schedule = None
        self.momentum_schedule = None

    def _build_momentum_network(self, task: tasks.ClassyTask) -> None:
        """
        Create the teacher: it is an exponential moving average of the student.
        """
        logging.info("Building momentum encoder")

        # Same architecture but do not apply stochastic depth
        # TODO: make drop_path_rate configurable for teacher
        task.config["MODEL"]["TRUNK"]["VISION_TRANSFORMERS"]["DROP_PATH_RATE"] = 0.0
        task.loss.momentum_teacher = build_model(
            task.config["MODEL"], task.config["OPTIMIZER"]
        )
        task.loss.momentum_teacher.to(task.device)

        # Restore an hypothetical checkpoint
        if task.loss.checkpoint is not None:
            task.loss.load_state_dict(task.loss.checkpoint)
        # Initialize from the model
        else:
            task_model = get_no_ddp_model(task.model)
            teacher_model = get_no_ddp_model(task.loss.momentum_teacher)
            teacher_model.load_state_dict(task_model.state_dict())

        # Setup SyncBN (useful for the XCiT)
        task.loss.momentum_teacher = nn.SyncBatchNorm.convert_sync_batchnorm(
            task.loss.momentum_teacher
        )
        task.loss.momentum_teacher = DistributedDataParallel(
            task.loss.momentum_teacher, device_ids=[task.device]
        )

        # no gradients for teacher model
        for p in task.loss.momentum_teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_momentum_network(self, task: tasks.ClassyTask) -> None:
        """
        EMA update
        Each teacher parameter becomes a weighted average of its old self and the
        newest student.
        """
        # Cosine schedule for the teacher momentum
        m = 1 - 0.5 * (1 - task.loss.loss_config.momentum) * (
            math.cos(math.pi * task.iteration / task.max_iteration) + 1
        )
        task.additional_log_data["dino_teacher_momentum"] = m

        task_model = get_no_ddp_model(task.model)
        teacher_model = get_no_ddp_model(task.loss.momentum_teacher)

        # EMA update for the teacher parameters
        for param_q, param_k in zip(
            task_model.parameters(), teacher_model.parameters()
        ):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    @torch.no_grad()
    def update_teacher_temperature(self, task: tasks.ClassyTask) -> None:
        """
        Update the teacher temperature
        """
        if self.teacher_temp_schedule is None:
            teacher_temp_min = task.loss.loss_config["teacher_temp_min"]
            teacher_temp_max = task.loss.loss_config["teacher_temp_max"]
            teacher_temp_warmup_iters = task.loss.loss_config[
                "teacher_temp_warmup_iters"
            ]
            self.teacher_temp_schedule = torch.cat(
                (
                    torch.linspace(
                        teacher_temp_min, teacher_temp_max, teacher_temp_warmup_iters
                    ),
                    torch.ones(max(0, task.max_iteration - teacher_temp_warmup_iters))
                    * teacher_temp_max,
                )
            )

        teacher_temp = self.teacher_temp_schedule[task.iteration].item()
        task.loss.teacher_temp = teacher_temp
        task.additional_log_data["dino_teacher_temp"] = teacher_temp

    @torch.no_grad()
    def on_forward(self, task: tasks.ClassyTask) -> None:
        """
        Forward pass with momentum network. We forward momentum teacher
        only on the large resolution crops.
        """

        # Create the momentum teacher and its center if this is the first forward of a run
        if task.loss.momentum_teacher is None:
            self._build_momentum_network(task)

        # Compute momentum teacher features
        im_k = [
            task.last_batch.sample["input"][i]
            for i in task.loss.loss_config["crops_for_teacher"]
        ]
        task.loss.teacher_output = task.loss.momentum_teacher(im_k)[0][-1]
        self.update_teacher_temperature(task)

    @torch.no_grad()
    def on_update(self, task: "tasks.ClassyTask") -> None:
        self._update_momentum_network(task)
