# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch.nn.utils as utils
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.tasks.classification_task import AmpType
from fairscale.nn import FullyShardedDataParallel


class GradClipHook(ClassyHook):
    """
    Hook executed on a backward pass that clips gradients such that their
    norm does not exceed a specific value. Dosovitskiy et al. found it
    to be critical for training vision transformers
    (https://arxiv.org/abs/2010.11929), but subsequent studies have been less
    clear about its importance. Gradient clipping configuration is set in
    config.MODEL.GRAD_CLIP
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, norm_type: Union[int, float, str], max_norm: Union[int, float]):
        super().__init__()
        self.norm_type = norm_type
        self.max_norm = max_norm

    def on_backward(self, task: tasks.ClassyTask) -> None:
        if task.amp_type == AmpType.PYTORCH:
            task.amp_grad_scaler.unscale_(task.optimizer)

        if isinstance(task.model, FullyShardedDataParallel):
            task.model.clip_grad_norm_(max_norm=self.max_norm, norm_type=self.norm_type)
        else:
            utils.clip_grad_norm_(
                task.model.parameters(),
                max_norm=self.max_norm,
                norm_type=self.norm_type,
            )
