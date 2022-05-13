# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from classy_vision.losses import ClassyLoss, register_loss
from fairscale.nn import FullyShardedDataParallel
from vissl.config import AttrDict
from vissl.models.model_helpers import get_no_ddp_model
from vissl.utils.distributed_utils import concat_all_gather


@register_loss("ibot_loss")
class IBOTLoss(ClassyLoss):
    """Wrapper around the loss of IBOT (https://arxiv.org/pdf/2111.07832.pdf)

    Allows to feed the correct configuration to IBOTCriterion class (below)
    which implement the IBOT loss entirely decoupled from VISSL
    """

    def __init__(self, loss_config: AttrDict):
        super().__init__()

        # iBOT criterion
        self.num_global_crops = loss_config["num_global_crops"]
        self.criterion = iBOTCriterion(
            out_dim=loss_config["out_dim"],
            patch_out_dim=loss_config["patch_out_dim"],
            ngcrops=self.num_global_crops,
            nlcrops=loss_config["num_local_crops"],
            warmup_teacher_temp=loss_config["warmup_teacher_temp"],
            teacher_temp=loss_config["teacher_temp"],
            warmup_teacher_temp2=loss_config["warmup_teacher_patch_temp"],
            teacher_temp2=loss_config["teacher_patch_temp"],
            warmup_teacher_temp_epochs=loss_config["warmup_teacher_temp_epochs"],
            nepochs=loss_config["num_epochs"],
            student_temp=loss_config["student_temp"],
            center_momentum=loss_config["center_momentum"],
            center_momentum2=loss_config["center_momentum2"],
            lambda1=loss_config["lambda1"],
            lambda2=loss_config["lambda2"],
            mim_start_epoch=loss_config["mim_start_epoch"],
        )

        # Attributes needed for the Hook
        self.current_epoch = 0
        self.teacher_momentum = loss_config["teacher_momentum"]

        # Momentum teacher related attributes
        self.momentum_teacher = None
        self.checkpoint = None
        self.teacher_output = None
        self.teacher_temp = None

    def set_current_epoch(self, epoch: int):
        self.current_epoch = epoch

    @classmethod
    def from_config(cls, loss_config: AttrDict):
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
                "criterion": self.criterion.state_dict(),
                "teacher": self.momentum_teacher.local_state_dict(),
                "teacher_meta": self.momentum_teacher.local_metadata_dict(),
            }
        else:
            return {
                "criterion": self.criterion.state_dict(),
                "teacher": self.momentum_teacher.state_dict(),
            }

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        # If the teacher not not been yet allocated, store it to load it
        # once the momentum teacher is available
        if self.momentum_teacher is None:
            self.checkpoint = state_dict
            logging.info("Storing the checkpoint for later use")
            return

        # If the encoder has been allocated, use the normal PyTorch restoration
        logging.info("Restoring checkpoint")
        if isinstance(self.momentum_teacher, FullyShardedDataParallel):
            self.momentum_teacher.load_local_state_dict(state_dict["teacher"])
        else:
            momentum_teacher = get_no_ddp_model(self.momentum_teacher)
            teacher_state_dict = {
                k.replace("module.", ""): v for k, v in state_dict["teacher"].items()
            }
            momentum_teacher.load_state_dict(teacher_state_dict)
        self.criterion.load_state_dict(state_dict["criterion"])

    def forward(self, output: List[List[torch.Tensor]], target: torch.Tensor):
        error_message = "Two outputs expected: student global and local views"
        assert len(output) == 2, error_message
        student_global_out, student_local_out = output
        teacher_global_out = self.teacher_output
        student_mask = target

        error_message = "Two outputs expected: class tokens and patch tokens"
        assert len(student_global_out) == 2, error_message
        assert len(teacher_global_out) == 2, error_message

        # Change the shape of the student mask to fit the need for
        # the iBOT official criterion: list of tensors of shape (B, H, W)
        student_mask_list = list(student_mask.chunk(self.num_global_crops))

        # Compute the loss and its sub-components (returns a dict)
        loss = self.criterion(
            student_global_out,
            teacher_global_out,
            student_local_cls=student_local_out[0],
            student_mask=student_mask_list,
            epoch=self.current_epoch,
        )

        # Computing additional statistics
        loss["acc_cls"] = self._extract_cls_token_accuracy(
            student_global_out[0], teacher_global_out[0]
        )
        loss["acc_ptc"] = self._extract_ptc_token_accuracy(
            student_global_out[1], teacher_global_out[1], student_mask
        )

        # Return the enriched loss
        return loss

    def _extract_cls_token_accuracy(
        self, teacher_global_cls_out: torch.Tensor, student_global_cls_out: torch.Tensor
    ):
        """
        Check if class token predictions match across global views:
        - masked vs unmasked
        - crop 1 vs crop 2
        - student vs teacher

        Args:
            teacher_global_cls_out: (batch_size * num_global_crops, out_dim)
            student_global_cls_out: (batch_size * num_global_crops, out_dim)
        """
        t_probs = teacher_global_cls_out.chunk(self.num_global_crops)
        s_probs = student_global_cls_out.chunk(self.num_global_crops)
        pred1 = concat_all_gather(t_probs[0].max(dim=1)[1])
        pred2 = concat_all_gather(s_probs[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        return acc.item()

    def _extract_ptc_token_accuracy(
        self, teacher_global_patches, student_global_patches, student_mask
    ):
        """
        Check if class token predictions match across local views:
        - on masked patches
        - student vs teacher

        Args:
            teacher_global_patches: (batch_size * num_global_crops, feat_map_size, out_dim)
            student_global_patches: (batch_size * num_global_crops, feat_map_size, out_dim)
            student_mask: (batch_size * num_global_crops, H, W)
        """

        # TODO (IBOT) - provide faster implementation where we avoid all_gather masked info
        teacher_pred = teacher_global_patches.max(dim=-1)[1].view(-1)
        student_pred = student_global_patches.max(dim=-1)[1].view(-1)
        teacher_pred = concat_all_gather(teacher_pred)
        student_pred = concat_all_gather(student_pred)
        student_mask = concat_all_gather(student_mask.view(-1))

        teacher_pred = torch.masked_select(teacher_pred, student_mask)
        student_pred = torch.masked_select(student_pred, student_mask)
        acc = (teacher_pred == student_pred).sum() / student_mask.sum().clamp(min=1.0)
        return acc.item()

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)


class iBOTCriterion(nn.Module):
    """Loss used in iBOT (https://arxiv.org/pdf/2111.07832.pdf)

    Mostly copy-pasted from the official iBOT repository:
    https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/main_ibot.py#L478
    """

    def __init__(
        self,
        out_dim: int,
        patch_out_dim: int,
        ngcrops: int,
        nlcrops: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_temp2: float,
        teacher_temp2: float,
        warmup_teacher_temp_epochs: int,
        nepochs: int,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=1.0,
        lambda2=1.0,
        mim_start_epoch=0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.teacher_temp2_schedule = (
            np.concatenate(
                (
                    np.linspace(
                        warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2,
                )
            )
            if mim_start_epoch == 0
            else np.concatenate(
                (
                    np.ones(mim_start_epoch) * warmup_teacher_temp2,
                    np.linspace(
                        warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch)
                    * teacher_temp2,
                )
            )
        )

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_local_cls: torch.Tensor,
        student_mask: List[torch.Tensor],
        epoch: int,
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.

        Args:
            student_output:     global views class token + feature map
            teacher_output:     global views class token + feature map
            student_local_cls:  local views class token only
            student_mask:       mask used for the global view
            epoch (int):        current epoch of training
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                # If same view: don't distill the class token but the patch tokens
                # for the masked image modeling criterion part of iBOT
                if v == q:
                    loss2 = torch.sum(
                        -teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1,
                    )
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(
                        dim=-1
                    ).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                # If different views: distill the class tokens as
                # it encourages to push toward recognizing same content
                else:
                    loss1 = torch.sum(
                        -teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1),
                        dim=-1,
                    )
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = {
            "cls": total_loss1,
            "patch": total_loss2,
            "loss": total_loss1 + total_loss2,
        }
        self.update_center(teacher_cls, teacher_patch)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum
        )

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2
        )
