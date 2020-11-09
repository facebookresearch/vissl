# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union


import torch.nn.utils as utils
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook


class GradClipHook(ClassyHook):
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, norm_type: Union[int, float, str],
                 max_norm: Union[int, float]):
        super().__init__()
        self.norm_type = norm_type
        self.max_norm = max_norm

    def on_backward(self, task: tasks.ClassyTask) -> None:
        utils.clip_grad_norm_(task.model.parameters(),
                              max_norm=self.max_norm, norm_type=self.norm_type)
