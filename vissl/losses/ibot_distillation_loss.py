# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint
from typing import List, Optional

import torch
import torch.nn.functional as F
from classy_vision.losses import ClassyLoss, register_loss
from vissl.config import AttrDict


@register_loss("ibot_distillation_loss")
class IBOTDistillationLoss(ClassyLoss):
    def __init__(self, loss_config: AttrDict):
        super().__init__()
        self.loss_config = loss_config
        self.swapped_assignment = self.loss_config.swapped_assignment
        self.use_teacher_prototypes = self.loss_config.use_teacher_prototypes
        self.num_global_crops = self.loss_config.num_global_crops
        self.student_num_crops = self.loss_config.student_num_crops
        self.student_temp = self.loss_config.student_temp
        self.teacher_temp = self.loss_config.teacher_temp
        self.teacher_patch_temp = self.loss_config.teacher_patch_temp
        self.lambda1 = self.loss_config.lambda1
        self.lambda2 = self.loss_config.lambda2

        # Teacher prototype probabilities that will be used to compute the
        # assignments for the student model crops prototype scores
        self.teacher_scores: Optional[torch.Tensor] = None

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        return cls(loss_config)

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)

    def forward(self, student_scores: List[List[torch.Tensor]], target: torch.Tensor):

        # Extract local and global views
        error_message = "Two outputs expected: student global and local views"
        assert len(student_scores) == 2, error_message
        student_global_out, student_local_out = student_scores
        teacher_global_out = self.teacher_scores

        # Extract class token and positional tokens
        error_message = "Two outputs expected: class tokens and patch tokens"
        assert len(student_global_out) == 2, error_message
        assert len(teacher_global_out) == 2, error_message
        assert len(student_local_out) == 1

        # Change the shape of the student mask to fit the need for
        # the iBOT official criterion: list of tensors of shape (B, H, W)
        student_mask = target
        student_mask_list = list(student_mask.chunk(self.num_global_crops))

        # Compute the loss and its sub-components (returns a dict)
        loss = self.criterion(
            student_global_out,
            teacher_global_out,
            student_local_cls=student_local_out[0],
            student_mask=student_mask_list,
        )
        return loss["loss"]

    def criterion(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_local_cls: torch.Tensor,
        student_mask: List[torch.Tensor],
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.

        Args:
            student_output:     global views class token + feature map
            teacher_output:     global views class token + feature map
            student_local_cls:  local views class token only
            student_mask:       mask used for the global view
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for student
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.student_num_crops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.num_global_crops)

        # Teacher sharpening
        teacher_cls_c = F.softmax(teacher_cls / self.teacher_temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.num_global_crops)
        teacher_patch_c = F.softmax(teacher_patch / self.teacher_patch_temp, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.num_global_crops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):

                # Distillation of the patch tokens for the masked image modeling
                # criterion part of iBOT (only if same view)
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

                # Distill the class token to recognize same content if same view, if:
                # - student is on a small crop
                # - student is on a big crop != from teacher and swapped_assignment
                # - student is on a big crop == from teacher and not swapped_assignment
                distill_class_token = False
                if v >= self.num_global_crops:
                    distill_class_token = True
                elif self.swapped_assignment and v != q:
                    distill_class_token = True
                elif not self.swapped_assignment and v == q:
                    distill_class_token = True

                # Distill the class tokens to encourage to push toward
                # recognizing same content
                if distill_class_token:
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
        return total_loss
