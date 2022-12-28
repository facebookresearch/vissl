# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint
from typing import List, Optional, Union

import torch
import torch.nn.functional as functional
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict


@register_loss("dino_distillation_loss")
class DINODistillationLoss(ClassyLoss):
    """
    This loss allows to distill a DINO model
    """

    def __init__(self, loss_config: AttrDict):
        super().__init__()
        self.loss_config = loss_config
        self.criterion = DINODistillationCriterion(
            student_num_crops=self.loss_config.student_num_crops,
            teacher_num_crops=self.loss_config.teacher_num_crops,
            student_temperature=self.loss_config.student_temperature,
            teacher_temperature=self.loss_config.teacher_temperature,
            swapped_assignment=self.loss_config.swapped_assignment,
        )

        # Teacher prototype scores that will be used to compute the
        # assignments for the student model crops prototype scores
        self.teacher_scores: Optional[torch.Tensor] = None

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        return cls(loss_config)

    def forward(
        self,
        student_scores: Union[List[torch.Tensor], torch.Tensor],
        target: torch.Tensor,
    ):
        if isinstance(student_scores, list):
            student_scores = student_scores[0]
        return self.criterion(student_scores, self.teacher_scores)

    @property
    def teacher_num_crops(self):
        return self.criterion.teacher_num_crops

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "criterion": repr(self.criterion),
        }
        return pprint.pformat(repr_dict, indent=2)


class DINODistillationCriterion(nn.Module):
    """
    This loss allows to distill a SwAV model
    """

    def __init__(
        self,
        teacher_temperature: float,
        student_temperature: float,
        student_num_crops: int,
        teacher_num_crops: int,
        swapped_assignment: bool,
    ):
        super().__init__()
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.student_num_crops = student_num_crops
        self.teacher_num_crops = teacher_num_crops
        self.swapped_assignment = swapped_assignment

    def forward(
        self,
        student_scores: torch.Tensor,
        teacher_scores: torch.Tensor,
    ):
        student_out = student_scores / self.student_temperature
        student_out = student_out.chunk(self.student_num_crops)
        teacher_out = functional.softmax(
            teacher_scores.detach() / self.teacher_temperature, dim=-1
        )
        teacher_out = teacher_out.chunk(self.teacher_num_crops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if self.swapped_assignment:
                    # We skip cases where student and teacher operate on the same view
                    # in case we do swapped assignments
                    if v == iq:
                        continue
                else:
                    # Else we skip the cross assignments between big crops
                    if v < self.teacher_num_crops and v != iq:
                        continue

                loss = torch.sum(
                    -q * functional.log_softmax(student_out[v], dim=-1), dim=-1
                )
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "student_num_crops": self.student_num_crops,
            "teacher_num_crops": self.teacher_num_crops,
            "student_temperature": self.student_temperature,
            "teacher_temperature": self.teacher_temperature,
            "swapped_assignment": self.swapped_assignment,
        }
        return pprint.pformat(repr_dict, indent=2)
