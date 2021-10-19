# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

import torch
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from torch import nn
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
        the momentum teacher, optionally normalize the last layer and updating the teacher
        temperature. At the beginning of training i.e. after
        1st forward call, the encoder is contructed and updated.
        """
        super().__init__()
        self.teacher_temp_schedule = None
        self.momentum_schedule = None

    def _build_momentum_network(self, task: tasks.ClassyTask) -> None:
        """
        Create the teacher: it is an exponential moving average of the student.
        """
        logging.info("Building momentum encoder")

        # - same architecture but do not apply stochastic depth
        # TODO: make drop_path_rate configurable for teacher
        task.config["MODEL"]["TRUNK"]["VISION_TRANSFORMERS"]["DROP_PATH_RATE"] = 0
        task.loss.momentum_teacher = build_model(
            task.config["MODEL"], task.config["OPTIMIZER"]
        )
        task.loss.momentum_teacher.to(task.device)

        # no gradients for teacher model
        for p in task.loss.momentum_teacher.parameters():
            p.requires_grad = False

        # Restore an hypothetical checkpoint
        if task.loss.checkpoint is not None:
            task.loss.load_state_dict(task.loss.checkpoint)
        # Initialize from the model
        else:
            task_model = get_no_ddp_model(task.model)
            teacher_model = get_no_ddp_model(task.loss.momentum_teacher)
            teacher_model.load_state_dict(task_model.state_dict())

        task.loss.momentum_teacher = nn.SyncBatchNorm.convert_sync_batchnorm(
            task.loss.momentum_teacher
        )

        # Newer PyTorch versions throw error for using DDP with a model that
        # has zero trainable parameters
        # if get_world_size() > 1:
        #     task.loss.momentum_teacher = init_distributed_data_parallel_model(
        #         task.loss.momentum_teacher
        #     )

    @torch.no_grad()
    def _update_momentum_network(self, task: tasks.ClassyTask) -> None:
        """
        EMA update
        Each teacher parameter becomes a weighted average of its old self and the
        newest student.
        """
        m = 1 - 0.5 * (1 - task.loss.loss_config.momentum) * (
            math.cos(math.pi * task.iteration / task.max_iteration) + 1
        )

        task_model = get_no_ddp_model(task.model)
        teacher_model = get_no_ddp_model(task.loss.momentum_teacher)

        # EMA update for the teacher parameters
        for param_q, param_k in zip(
            task_model.parameters(), teacher_model.parameters()
        ):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    @torch.no_grad()
    def get_teacher_temperature(self, task: tasks.ClassyTask) -> None:
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
        task.loss.teacher_temp = self.teacher_temp_schedule[task.iteration].item()

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
        self.get_teacher_temperature(task)

    @torch.no_grad()
    def on_update(self, task: "tasks.ClassyTask") -> None:
        self._update_momentum_network(task)
        self.normalize_last_layer(task)

    @torch.no_grad()
    def on_start(self, task: "tasks.ClassyTask") -> None:
        """
        Optionally normalize prototypes
        """
        self.normalize_last_layer(task)

    @torch.no_grad()
    def normalize_last_layer(self, task: "tasks.ClassyTask") -> None:
        """
        Optionally normalize prototypes
        """
        if not task.config.LOSS["dino_loss"].normalize_last_layer:
            return
        try:
            for j in range(task.model.heads[0].nmb_heads):
                w = getattr(
                    task.model.heads[0], "prototypes" + str(j)
                ).weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                getattr(task.model.heads[0], "prototypes" + str(j)).weight.copy_(w)
        except AttributeError:
            # TODO (mathildecaron): don't use getattr
            for j in range(task.model.module.heads[0].nmb_heads):
                w = getattr(
                    task.model.module.heads[0], "prototypes" + str(j)
                ).weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                getattr(task.model.module.heads[0], "prototypes" + str(j)).weight.copy_(
                    w
                )
        if task.loss.momentum_teacher is not None:
            try:
                for j in range(task.loss.momentum_teacher.heads[0].nmb_heads):
                    w = getattr(
                        task.loss.momentum_teacher.heads[0], "prototypes" + str(j)
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(
                        task.loss.momentum_teacher.heads[0], "prototypes" + str(j)
                    ).weight.copy_(w)
            except AttributeError:
                for j in range(task.loss.momentum_teacher.module.heads[0].nmb_heads):
                    w = getattr(
                        task.loss.momentum_teacher.module.heads[0],
                        "prototypes" + str(j),
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(
                        task.loss.momentum_teacher.module.heads[0],
                        "prototypes" + str(j),
                    ).weight.copy_(w)
