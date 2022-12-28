# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint
from typing import List, Optional

import torch
import torch.nn.functional as functional
from classy_vision.losses import ClassyLoss, register_loss
from vissl.config import AttrDict


@register_loss("msn_distillation_loss")
class MSNDistillationLoss(ClassyLoss):
    def __init__(self, loss_config: AttrDict):
        super().__init__()
        self.loss_config = loss_config
        self.swapped_assignment = self.loss_config.swapped_assignment
        self.student_temperature = self.loss_config.student_temperature
        self.student_num_crops = self.loss_config.student_num_crops
        self.teacher_num_crops = self.loss_config.teacher_num_crops

        # Teacher prototype probabilities that will be used to compute the
        # assignments for the student model crops prototype scores
        self.teacher_probs: Optional[torch.Tensor] = None

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        return cls(loss_config)

    def forward(self, student_scores: List[torch.Tensor], *args, **kwargs):
        if isinstance(student_scores, list):
            student_scores = student_scores[0]
        return self.criterion(student_scores, self.teacher_probs)

    def criterion(
        self,
        student_scores: torch.Tensor,
        teacher_probs: torch.Tensor,
    ):
        student_out = student_scores / self.student_temperature
        student_out = student_out.chunk(self.student_num_crops)
        teacher_out = teacher_probs.chunk(self.teacher_num_crops)

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
            "student_temperature": self.student_temperature,
        }
        return pprint.pformat(repr_dict, indent=2)
