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
from fairscale.nn import FullyShardedDataParallel
from vissl.config import AttrDict
from vissl.models.model_helpers import get_no_ddp_model


@register_loss("dino_loss")
class DINOLoss(ClassyLoss):
    def __init__(self, loss_config: AttrDict):
        super().__init__()

        # Store loss configuration for DINOHook to access
        self.teacher_momentum = loss_config["momentum"]
        self.teacher_temp_min = loss_config["teacher_temp_min"]
        self.teacher_temp_max = loss_config["teacher_temp_max"]
        self.teacher_temp_warmup_iters = loss_config["teacher_temp_warmup_iters"]
        self.crops_for_teacher = loss_config["crops_for_teacher"]
        self.loss_config = loss_config

        # Momentum teacher related attributes
        self.momentum_teacher = None
        self.checkpoint = None
        self.teacher_output = None
        self.teacher_temp = None

        # Loss center related attributes
        self.is_distributed = is_distributed_training_run()
        self.use_gpu = get_cuda_device_index() > -1
        self.register_buffer("center", torch.zeros(1, loss_config["output_dim"]))

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

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return the state dictionary for the loss:
        - For DDP model, we rely on the default implementation of PyTorch
        - For FSDP model, we only return a shard of the teacher model and
          each rank will do the same, limiting memory consumption
        """
        if isinstance(self.momentum_teacher, FullyShardedDataParallel):
            return {
                "center": self.center.data,
                "teacher": self.momentum_teacher.local_state_dict(),
                "teacher_meta": self.momentum_teacher.local_metadata_dict(),
            }
        else:
            return super().state_dict(destination, prefix, keep_vars)

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

        # If the encoder has been allocated, use the normal pytorch restoration
        logging.info("Restoring checkpoint")
        if isinstance(self.momentum_teacher, FullyShardedDataParallel):
            self.momentum_teacher.load_local_state_dict(state_dict["teacher"])
            self.center.copy_(state_dict["center"])
        else:
            teacher_model = get_no_ddp_model(self.momentum_teacher)
            # teacher params
            sd = {
                x.replace("momentum_teacher.module.", ""): state_dict[x]
                for x in state_dict
                if x.find("momentum_teacher") != -1
            }
            sd = {x.replace("momentum_teacher.", ""): sd[x] for x in sd}
            teacher_model.load_state_dict(sd)

            # center value
            non_teacher_sd = {
                x: state_dict[x] for x in state_dict if x.find("momentum_teacher") == -1
            }
            assert (
                len(non_teacher_sd) == 1
            ), f"State dict for loss has multiple non teacher keys: {non_teacher_sd.keys()}"
            self.center.copy_(state_dict["center"])
            logging.info("Loaded the center in dino loss...")

    @torch.no_grad()
    def update_center(self):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(self.teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(self.teacher_output) * get_world_size())

        # ema update
        center_momentum = self.loss_config.ema_center
        self.center = self.center * center_momentum + batch_center * (
            1 - center_momentum
        )

    def forward(self, output: List[torch.Tensor], *args, **kwargs):
        student_out = output[-1] / self.loss_config["student_temp"]
        student_out = student_out.chunk(self.loss_config["num_crops"])

        # teacher centering and sharpening
        teacher_out = F.softmax(
            (self.teacher_output - self.center) / self.teacher_temp, dim=-1
        )
        teacher_out = teacher_out.detach().chunk(
            len(self.loss_config["crops_for_teacher"])
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
