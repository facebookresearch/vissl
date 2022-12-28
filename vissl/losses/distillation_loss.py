# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as functional
from classy_vision.losses import ClassyLoss, register_loss
from vissl.config.attr_dict import AttrDict
from vissl.losses.cross_entropy_multiple_output_single_target import SmoothCrossEntropy


class DistillationType(Enum):
    MSE = 0
    KL_DIVERGENCE = 1
    CROSS_ENTROPY = 2


class DistillationCriterion(nn.Module):
    def __init__(
        self,
        soft_target_alpha: float,
        temperature: float = 1.0,
        loss_type: DistillationType = DistillationType.KL_DIVERGENCE,
    ):
        super().__init__()
        self.hard_target_alpha = 1 - soft_target_alpha
        self.soft_target_alpha = soft_target_alpha
        self.temperature = temperature
        self.loss_type = loss_type
        self.smooth_cross_entropy = SmoothCrossEntropy()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
    ):
        teacher_loss = self._get_teacher_loss(student_logits, teacher_logits)
        if self.hard_target_alpha > 0.0:
            target_loss = self.smooth_cross_entropy(student_logits, target)
            return (
                self.hard_target_alpha * target_loss
                + self.soft_target_alpha * teacher_loss
            )
        else:
            return teacher_loss

    def _get_teacher_loss(self, student_logits, teacher_logits):
        if self.loss_type == DistillationType.MSE:
            return functional.mse_loss(student_logits, teacher_logits)
        elif self.loss_type == DistillationType.KL_DIVERGENCE:
            return self._get_kl_divergence_penalty(student_logits, teacher_logits) * (
                self.temperature**2
            )
        elif self.loss_type == DistillationType.CROSS_ENTROPY:
            return self._get_cross_entropy_penalty(student_logits, teacher_logits) * (
                self.temperature**2
            )
        return torch.tensor(0.0, device=student_logits.device)

    def _get_kl_divergence_penalty(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ):
        # Computes KL(teacher || student) not KL(student || teacher)
        # Note: PyTorch KL Divergence expects log probabilities
        student_log_probs = functional.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = functional.softmax(teacher_logits / self.temperature, dim=-1)
        return functional.kl_div(
            student_log_probs, teacher_probs, reduction="batchmean", log_target=False
        )

    def _get_cross_entropy_penalty(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ):
        # Cross entropy against the target distribution of teacher
        student_log_probs = functional.log_softmax(
            student_logits / self.temperature, dim=-1
        )
        teacher_probs = functional.softmax(teacher_logits / self.temperature, dim=-1)
        return -torch.mean(torch.sum(student_log_probs * teacher_probs, dim=-1))


@register_loss("distillation_loss")
class DistillationLoss(ClassyLoss):
    def __init__(self, soft_target_alpha: float, temperature: float, loss_type: str):
        super().__init__()
        self.criterion = DistillationCriterion(
            soft_target_alpha=soft_target_alpha,
            temperature=temperature,
            loss_type=DistillationType[loss_type.upper()],
        )
        self.teacher_logits = None

    @classmethod
    def from_config(cls, loss_config: AttrDict) -> "DistillationLoss":
        return cls(
            soft_target_alpha=loss_config["soft_target_alpha"],
            temperature=loss_config["temperature"],
            loss_type=loss_config.get("loss_type", "kl_divergence"),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        return self.criterion(logits, self.teacher_logits, target)
