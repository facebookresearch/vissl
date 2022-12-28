# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.hooks.hook_utils import MomentumTeacherInLossHook


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

    @staticmethod
    def _build_momentum_network(task: tasks.ClassyTask) -> None:
        # Same architecture as student but do not apply stochastic depth
        task.config["MODEL"]["TRUNK"]["VISION_TRANSFORMERS"]["DROP_PATH_RATE"] = 0.0
        MomentumTeacherInLossHook.build_momentum_network(task, config=task.config)

    @torch.no_grad()
    def update_teacher_temperature(self, task: tasks.ClassyTask) -> None:
        """
        Update the teacher temperature
        """
        if self.teacher_temp_schedule is None:
            teacher_temp_min = task.loss.teacher_temp_min
            teacher_temp_max = task.loss.teacher_temp_max
            teacher_temp_warmup_iters = task.loss.teacher_temp_warmup_iters

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
        im_k = [task.last_batch.sample["input"][i] for i in task.loss.crops_for_teacher]
        task.loss.teacher_output = task.loss.momentum_teacher(im_k)[0][-1]
        self.update_teacher_temperature(task)

    @torch.no_grad()
    def on_update(self, task: "tasks.ClassyTask") -> None:
        MomentumTeacherInLossHook.update_momentum_network(
            task,
            init_teacher_momentum=task.loss.teacher_momentum,
            with_cosine_schedule=True,
        )
