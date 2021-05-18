# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from classy_vision.generic.distributed_util import (
    get_cuda_device_index,
    get_world_size,
    is_distributed_training_run,
)
from classy_vision.losses import ClassyLoss, register_loss
from vissl.config import AttrDict


@register_loss("dino_loss")
class DINOLoss(ClassyLoss):
    def __init__(self, loss_config: AttrDict):
        super().__init__()
        self.loss_config = loss_config
        self.momentum_teacher = None
        self.checkpoint = None
        self.teacher_output = None
        self.teacher_temp = None
        self.is_distributed = is_distributed_training_run()
        self.use_gpu = get_cuda_device_index() > -1
        self.center = None

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates DINOLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            DINOLoss instance.
        """
        return cls(loss_config)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Restore the loss state given a checkpoint

        Args:
            state_dict (serialized via torch.save)
        """

        # If the encoder has been allocated, use the normal pytorch restoration
        if self.momentum_teacher is None:
            self.checkpoint = state_dict
            logging.info("Storing the checkpoint for later use")
        else:
            logging.info("Restoring checkpoint")
            super().load_state_dict(state_dict, *args, **kwargs)

    @torch.no_grad()
    def update_center(self):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(self.teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(self.teacher_output) * get_world_size())

        # ema update
        m = self.loss_config.ema_center
        self.center = self.center * m + batch_center * (1 - m)

    def forward(self, output: List[torch.Tensor], *args, **kwargs):
        student_out = output[-1] / self.loss_config.student_temp
        student_out = student_out.chunk(self.loss_config.num_crops)

        # teacher centering and sharpening
        teacher_out = F.softmax(
            (self.teacher_output - self.center) / self.teacher_temp, dim=-1
        )
        teacher_out = teacher_out.detach().chunk(
            len(self.loss_config.crops_for_teacher)
        )

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        self.update_center()
        return total_loss

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)
