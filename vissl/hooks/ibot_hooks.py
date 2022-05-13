# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.hooks.hook_utils import MomentumTeacherInLossHook


class IBOTHook(ClassyHook):
    """Hook used for the implementation of iBOT (https://arxiv.org/pdf/2111.07832.pdf)
    - Handles the creation of the EMA teacher
    - Handles the forward of the EMA teacher
    - Handles the configuration of the DINO Loss
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop

    def __init__(self):
        super().__init__()
        self.teacher_temp_schedule = None
        self.momentum_schedule = None

    @staticmethod
    def _build_momentum_network(task: tasks.ClassyTask) -> None:
        # Same architecture as student but do not apply stochastic depth
        task.config["MODEL"]["TRUNK"]["VISION_TRANSFORMERS"]["DROP_PATH_RATE"] = 0.0
        MomentumTeacherInLossHook.build_momentum_network(task, config=task.config)

    @torch.no_grad()
    def on_forward(self, task: tasks.ClassyTask) -> None:
        """
        Forward pass with momentum network. We forward momentum teacher
        only on the large resolution crops.
        """

        # Create the momentum teacher and its center if this is the first forward of a run
        if task.loss.momentum_teacher is None:
            self._build_momentum_network(task)

        # From the student inputs, only keep the global views
        # - ignore the local views
        # - ignore the mask for the global views
        last_student_input = task.last_batch.sample["input"]
        teacher_input = {"global_views": last_student_input["global_views"]}

        # Compute momentum teacher features
        teacher_output = task.loss.momentum_teacher(teacher_input)
        task.loss.teacher_output = teacher_output[0]
        task.loss.set_current_epoch(task.train_phase_idx)

    @torch.no_grad()
    def on_update(self, task: "tasks.ClassyTask") -> None:
        MomentumTeacherInLossHook.update_momentum_network(
            task,
            init_teacher_momentum=task.loss.teacher_momentum,
            with_cosine_schedule=True,
        )
