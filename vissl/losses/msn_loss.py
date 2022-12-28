# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint
from typing import List

import torch
from classy_vision.generic.distributed_util import (
    get_cuda_device_index,
    is_distributed_training_run,
)
from classy_vision.losses import ClassyLoss, register_loss
from fairscale.nn import FullyShardedDataParallel
from vissl.config import AttrDict
from vissl.models.model_helpers import get_no_ddp_model


@register_loss("msn_loss")
class MSNLoss(ClassyLoss):
    """Loss used in MSN (https://arxiv.org/pdf/2204.07141.pdf)

    Args:
        temperature (float): temperature
        teacher_momentum (float): controls EMA of teacher
    """

    def __init__(self, temperature: float, teacher_momentum: float):
        super().__init__()

        # Store loss configuration for DINOHook to access
        self.temperature = temperature
        self.teacher_momentum = teacher_momentum
        self.teacher_temp_min = temperature
        self.teacher_temp_max = temperature
        self.teacher_temp_warmup_iters = 0
        self.crops_for_teacher = [0]

        # Momentum teacher related attributes
        self.momentum_teacher = None
        self.checkpoint = None
        self.teacher_output = None
        self.teacher_temp = None
        self.is_distributed = is_distributed_training_run()
        self.use_gpu = get_cuda_device_index() > -1

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates MSNLoss from configuration.
        """
        # TODO - add missing argument to configure loss
        return cls(
            temperature=loss_config.temperature,
            teacher_momentum=loss_config.momentum,
        )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return the state dictionary for the loss:
        - For DDP model, we rely on the default implementation of PyTorch
        - For FSDP model, we only return a shard of the teacher model and
          each rank will do the same, limiting memory consumption
        """
        if isinstance(self.momentum_teacher, FullyShardedDataParallel):
            return {
                "teacher": self.momentum_teacher.local_state_dict(),
                "teacher_meta": self.momentum_teacher.local_metadata_dict(),
            }
        else:
            return {"teacher": self.momentum_teacher.state_dict()}

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Restore the loss state given a checkpoint

        Args:
            state_dict (serialized via torch.save)
        """

        # If the teacher not not been yet allocated, store it to load it
        # once the momentum teacher is available
        if self.momentum_teacher is None:
            self.checkpoint = state_dict
            logging.info("Storing the checkpoint for later use")
            return

        # If the momentum teacher is already allocated, use the normal
        # pytorch checkpoint restoration (load_state_dict)
        logging.info("Restoring checkpoint")
        if isinstance(self.momentum_teacher, FullyShardedDataParallel):
            self.momentum_teacher.load_local_state_dict(state_dict["teacher"])
            self.center.copy_(state_dict["center"])
        else:
            teacher_model = get_no_ddp_model(self.momentum_teacher)
            sd = {
                x.replace("momentum_teacher.module.", ""): state_dict[x]
                for x in state_dict
                if x.find("momentum_teacher") != -1
            }
            sd = {x.replace("momentum_teacher.", ""): sd[x] for x in sd}
            teacher_model.load_state_dict(sd)

    def forward(self, output: List[torch.Tensor], *args, **kwargs):

        # Retrieve student and teacher outputs
        student_out = output[-1]
        teacher_out = self.teacher_output

        # TODO - implement the correct loss of MSN
        loss = (student_out[: teacher_out.shape[0]] - teacher_out).sum() ** 2
        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "temperature": self.temperature,
            "teacher_momentum": self.teacher_momentum,
        }
        return pprint.pformat(repr_dict, indent=2)
