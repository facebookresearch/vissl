# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pprint
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as functional
from classy_vision.generic.distributed_util import (
    all_reduce_max,
    get_cuda_device_index,
    get_rank,
    get_world_size,
)
from classy_vision.losses import ClassyLoss, register_loss
from fvcore.common.file_io import PathManager
from torch import nn
from vissl.config import AttrDict
from vissl.losses.distibuted_sinkhornknopp import distributed_sinkhornknopp


@register_loss("swav_distillation_loss")
class SwAVDistillationLoss(ClassyLoss):
    """
    This loss allows to distill a SwAV model
    """

    def __init__(self, loss_config: AttrDict):
        super().__init__()
        self.loss_config = loss_config
        self.queue_start_iter = self.loss_config.get("queue_start_iter", 0)
        self.use_teacher_prototypes = self.loss_config.get(
            "use_teacher_prototypes", False
        )
        self.use_student_prototypes = self.loss_config.get(
            "use_student_prototypes", False
        )
        self.use_two_crops_for_teacher = self.loss_config.get(
            "use_two_crops_for_teacher", True
        )
        self.normalize_student_feats = self.loss_config.get(
            "normalize_student_feats", False
        )
        self.criterion = SwAVDistillationCriterion(
            temperature=self.loss_config.temperature,
            num_crops=self.loss_config.num_crops,
            num_iters=self.loss_config.num_iters,
            epsilon=self.loss_config.epsilon,
            use_double_prec=self.loss_config.use_double_precision,
            num_prototypes=self.loss_config.num_prototypes,
            output_dir=self.loss_config.output_dir,
            temp_hard_assignment_iters=self.loss_config.temp_hard_assignment_iters,
            local_queue_length=self.loss_config.local_queue_length,
            swapped_assignment=self.loss_config.swapped_assignment,
        )

        # Teacher prototype scores that will be used to compute the
        # assignments for the student model crops' prototype scores
        self.teacher_prototypes_scores: Optional[torch.Tensor] = None

        # Teacher prototypes that will be used to compute the
        # scores for each crops of the student model
        self.teacher_prototypes: Optional[torch.Tensor] = None

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        return cls(loss_config)

    def forward(
        self,
        student_outputs: Union[List[torch.Tensor], torch.Tensor],
        target: torch.Tensor,
    ):

        # Useful for heads such as DINO heads returning a list
        # Not useful for heads such as MLP returning a single tensor
        if isinstance(student_outputs, list):
            if self.use_student_prototypes:
                student_outputs = student_outputs[1]
            else:
                student_outputs = student_outputs[0]

        # Optional use of queue or normalization of student features
        self.criterion.use_queue = (
            self.criterion.local_queue_max_length > 0
            and self.criterion.num_iteration >= self.queue_start_iter
        )
        if self.normalize_student_feats:
            student_outputs = nn.functional.normalize(student_outputs, dim=-1, p=2)

        # Forward to distillation criterion
        loss = self.criterion(
            student_outputs,
            self.teacher_prototypes_scores,
            self.teacher_prototypes if self.use_teacher_prototypes else None,
        )
        self.criterion.num_iteration += 1
        if self.criterion.use_queue:
            self.criterion.add_to_teacher_scores_queue(self.teacher_prototypes_scores)
        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "epsilon": self.loss_config.epsilon,
            "use_double_precision": self.loss_config.use_double_precision,
            "temperature": self.loss_config.temperature,
            "num_prototypes": self.loss_config.num_prototypes,
            "num_crops": self.loss_config.num_crops,
            "nmb_sinkhornknopp_iters": self.loss_config.num_iters,
        }
        return pprint.pformat(repr_dict, indent=2)


class SwAVDistillationCriterion(nn.Module):
    """
    This loss allows to distill a SwAV model
    """

    def __init__(
        self,
        temperature: float,
        num_crops: int,
        num_iters: int,
        epsilon: float,
        use_double_prec: bool,
        num_prototypes: List[int],
        output_dir: str,
        temp_hard_assignment_iters: int,
        local_queue_length: int,
        swapped_assignment: bool,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_crops = num_crops
        self.nmb_sinkhornknopp_iters = num_iters
        self.epsilon = epsilon
        self.use_double_prec = use_double_prec
        self.num_prototypes = num_prototypes
        self.nmb_heads = len(self.num_prototypes)
        self.output_dir = output_dir
        self.temp_hard_assignment_iters = temp_hard_assignment_iters
        self.local_queue_max_length = local_queue_length
        self.swapped_assignment = swapped_assignment

        self.dist_rank = get_rank()
        self.world_size = get_world_size()
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.register_buffer("num_iteration", torch.zeros(1, dtype=int))
        self.use_gpu = get_cuda_device_index() > -1
        self.use_queue = False
        if self.local_queue_max_length > 0:
            self._init_teacher_scores_queue()

    def _init_teacher_scores_queue(self):
        self.local_queue_length = 0
        queue = torch.rand(self.local_queue_max_length, self.num_prototypes[0]) * 2 - 1
        self.register_buffer("local_queue", queue)

    def add_to_teacher_scores_queue(self, scores: torch.Tensor):
        """
        Append elements at the beginning of the local_queue
        """
        with torch.no_grad():
            bs = scores.shape[0]
            queue = self.local_queue
            queue[bs:] = queue[:-bs].clone()
            queue[:bs] = scores
            self.local_queue = queue
            self.local_queue_length = min(
                self.local_queue_length + bs, self.local_queue_max_length
            )

    def forward(
        self,
        logits: torch.Tensor,
        teacher_scores: torch.Tensor,
        teacher_prototypes: Optional[torch.Tensor],
    ):
        # Find the batch size and make sure inputs are correct
        assert logits.shape[0] % self.num_crops == 0
        batch_size = logits.shape[0] // self.num_crops
        assert teacher_scores.shape[0] % batch_size == 0
        num_crops_for_assign = teacher_scores.shape[0] // batch_size

        # Compute the scores based on the teacher prototypes
        if teacher_prototypes is not None:
            assert not teacher_prototypes.requires_grad
            scores = functional.linear(logits, teacher_prototypes)

        # Or take the logits of the student directly as scores if
        # the teacher prototypes are not provided as input
        else:
            message = f"Incompatible feature dimensions {logits.shape[1]} vs {teacher_scores.shape[1]}"
            assert logits.shape[1] == teacher_scores.shape[1], message
            scores = logits

        # Compute the loss toward each of the assignments
        loss = 0
        for i in range(num_crops_for_assign):
            teacher_crop_scores = teacher_scores[i * batch_size : (i + 1) * batch_size]

            # If just one crop to assign, decide whether or not we should map
            # the two crops of the student or the other one to the teacher
            if num_crops_for_assign == 1:
                start_crop = 1 if self.swapped_assignment else 0
                student_crop_ids = list(range(start_crop, self.num_crops))

            # If several crops to assign, either use swapped assignments or
            # non-swapped assignments
            elif self.swapped_assignment:
                student_crop_ids = [p for p in range(self.num_crops) if p != i]
            else:
                student_crop_ids = [i] + list(range(2, self.num_crops))

            loss += self.forward_to_teacher_crop(
                scores,
                teacher_crop_scores,
                batch_size=batch_size,
                student_crop_ids=student_crop_ids,
            )
        return loss / num_crops_for_assign

    def forward_to_teacher_crop(
        self,
        scores: torch.Tensor,
        teacher_scores: torch.Tensor,
        batch_size: int,
        student_crop_ids: List[int],
    ):
        # Compute the target assignments, taking crop_id as the features
        # used to compute the codes to which other crops will be mapped
        with torch.no_grad():
            assignments = self.compute_teacher_assignements(teacher_scores)

        # For each crop other than the one used as target assignment (id 0)
        # compute the cross entropy between the target assigment and the
        # softmax of the crop scores
        loss = 0
        for p in student_crop_ids:
            crop_scores = scores[batch_size * p : batch_size * (p + 1)]
            if self.use_double_prec:
                crop_scores = crop_scores.double() / np.float64(self.temperature)
            else:
                crop_scores = crop_scores / self.temperature
            minus_cross_entropies = torch.sum(
                assignments * self.log_softmax(crop_scores),
                dim=1,
                dtype=assignments.dtype,
            )
            loss -= torch.mean(minus_cross_entropies)

        # Average of the contribution of each crop (we don't want and
        # increase in the number of crop to impact the loss magnitude
        # and force us to update the LR)
        loss /= len(student_crop_ids)

        # Stop training if NaN appears and log the output to help debugging
        if torch.isnan(loss):
            logging.info(
                f"Infinite Loss or NaN. Loss value: {loss}, rank: {self.dist_rank}"
            )
            scores_output_file = os.path.join(
                self.output_dir, "rank" + str(self.dist_rank) + "_scores.pth"
            )
            assignments_out_file = os.path.join(
                self.output_dir, "rank" + str(self.dist_rank) + "_assignments.pth"
            )
            with PathManager.open(scores_output_file, "wb") as fwrite:
                torch.save(scores, fwrite)
            with PathManager.open(assignments_out_file, "wb") as fwrite:
                torch.save(assignments, fwrite)
            logging.info(f"Saved the scores matrix to: {scores_output_file}")
            logging.info(f"Saved the assignment matrix to: {assignments_out_file}")

        return loss

    def compute_teacher_assignements(self, teacher_scores: torch.Tensor):
        """
        Compute the teacher assignments toward which the student crops
        scores should converge
        """

        # In case we do not want to use Sink Horn Knopp and take the
        # teacher assignments as they are, return the softmax of
        # the teacher scores
        if self.nmb_sinkhornknopp_iters == 0:
            return torch.nn.functional.softmax(teacher_scores / self.epsilon, dim=-1)

        # Add teacher scores of the past stored in the queue
        batch_size = teacher_scores.shape[0]
        if self.use_queue:
            queue = self.local_queue[: self.local_queue_length]
            teacher_scores = torch.cat((teacher_scores, queue))

        # Divide by epsilon (which can be seen as a temperature which
        # helps to sharpen the distribution of the assignments)
        if self.use_double_prec:
            assignments = torch.exp(
                teacher_scores.double() / np.float64(self.epsilon)
            ).t()
            assignments = assignments.double()
        else:
            assignments = teacher_scores / self.epsilon
            # use the log-sum-exp trick for numerical stability.
            M = torch.max(assignments)
            all_reduce_max(M)
            assignments -= M
            assignments = torch.exp(assignments).t()

        # Apply sinkhornknopp algorithm to divide equally the
        # assignment to each of the prototypes
        assignments = distributed_sinkhornknopp(
            Q=assignments,
            hard_assignment=self.num_iteration < self.temp_hard_assignment_iters,
            world_size=self.world_size,
            num_iter=self.nmb_sinkhornknopp_iters,
            use_gpu=self.use_gpu,
            use_double_prec=self.use_double_prec,
        )

        # Extract the assignments of the current batch (discard queue
        # assignments used to stabilize sink-horn-knopp)
        assignments = assignments[:batch_size]
        return assignments

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "use_queue": self.use_queue,
            "local_queue_length": self.local_queue_max_length,
            "temperature": self.temperature,
            "num_prototypes": self.num_prototypes,
            "num_crops": self.num_crops,
            "nmb_sinkhornknopp_iters": self.nmb_sinkhornknopp_iters,
            "temp_hard_assignment_iters": self.temp_hard_assignment_iters,
        }
        return pprint.pformat(repr_dict, indent=2)
