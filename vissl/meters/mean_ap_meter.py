# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from classy_vision.generic.distributed_util import all_reduce_sum, gather_from_all
from classy_vision.meters import ClassyMeter, register_meter
from vissl.config import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.svm_utils.evaluate import get_precision_recall


@register_meter("mean_ap_meter")
class MeanAPMeter(ClassyMeter):
    """
    Meter to calculate mean AP metric for multi-label image classification task.

    Args:
        meters_config (AttrDict): config containing the meter settings

    meters_config should specify the num_classes
    """

    def __init__(self, meters_config: AttrDict):
        self.num_classes = meters_config.get("num_classes")
        self._total_sample_count = None
        self._curr_sample_count = None
        self.reset()

    @classmethod
    def from_config(cls, meters_config: AttrDict):
        """
        Get the AccuracyListMeter instance from the user defined config
        """
        return cls(meters_config)

    @property
    def name(self):
        """
        Name of the meter
        """
        return "mean_ap_manual_meter"

    @property
    def value(self):
        """
        Value of the meter globally synced. mean AP and AP for each class is returned
        """
        _, distributed_rank = get_machine_local_and_dist_rank()
        logging.info(
            f"Rank: {distributed_rank} Mean AP meter: "
            f"scores: {self._scores.shape}, target: {self._targets.shape}"
        )
        ap_matrix = torch.ones(self.num_classes, dtype=torch.float32) * -1
        # targets matrix = 0, 1, -1
        # unknown matrix = 0, 1 where 1 means that it's an unknown
        unknown_matrix = torch.eq(self._targets, -1.0).float().detach().numpy()
        for cls_num in range(self.num_classes):
            # compute AP only for classes that have at least one positive example
            num_pos = len(torch.where(self._targets[:, cls_num] == 1)[0])
            if num_pos == 0:
                continue
            P, R, score, ap = get_precision_recall(
                self._targets[:, cls_num].detach().numpy(),
                self._scores[:, cls_num].detach().numpy(),
                (unknown_matrix[:, cls_num] == 0).astype(float),
            )
            ap_matrix[cls_num] = ap[0]
        nonzero_indices = torch.nonzero(ap_matrix != -1)
        if nonzero_indices.shape[0] < self.num_classes:
            logging.info(
                f"{nonzero_indices.shape[0]} out of {self.num_classes} classes "
                "have meaningful average precision"
            )
        mean_ap = ap_matrix[nonzero_indices].mean().item()
        return {"mAP": mean_ap, "AP": ap_matrix}

    def gather_scores(self, scores: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            scores_gathered = gather_from_all(scores)
        else:
            scores_gathered = scores
        return scores_gathered

    def gather_targets(self, targets: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            targets_gathered = gather_from_all(targets)
        else:
            targets_gathered = targets
        return targets_gathered

    def sync_state(self):
        """
        Globally syncing the state of each meter across all the trainers.
        We gather scores, targets, total sampled
        """
        # Communications
        self._curr_sample_count = all_reduce_sum(self._curr_sample_count)
        self._scores = self.gather_scores(self._scores)
        self._targets = self.gather_targets(self._targets)

        # Store results
        self._total_sample_count += self._curr_sample_count

        # Reset values until next sync
        self._curr_sample_count.zero_()

    def reset(self):
        """
        Reset the meter
        """
        self._scores = torch.zeros(0, self.num_classes, dtype=torch.float32)
        self._targets = torch.zeros(0, self.num_classes, dtype=torch.int8)
        self._total_sample_count = torch.zeros(1)
        self._curr_sample_count = torch.zeros(1)

    def __repr__(self):
        return repr({"name": self.name, "value": self.value})

    def set_classy_state(self, state):
        """
        Set the state of meter
        """
        assert (
            self.name == state["name"]
        ), f"State name {state['name']} does not match meter name {self.name}"
        assert self.num_classes == state["num_classes"], (
            f"num_classes of state {state['num_classes']} "
            f"does not match object's num_classes {self.num_classes}"
        )

        # Restore the state -- correct_predictions and sample_count.
        self.reset()
        self._total_sample_count = state["total_sample_count"].clone()
        self._curr_sample_count = state["curr_sample_count"].clone()
        self._scores = state["scores"]
        self._targets = state["targets"]

    def get_classy_state(self):
        """
        Returns the states of meter
        """
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "scores": self._scores,
            "targets": self._targets,
            "total_sample_count": self._total_sample_count,
            "curr_sample_count": self._curr_sample_count,
        }

    def verify_target(self, target):
        """
        Verify that the target contains {-1, 0, 1} values only
        """
        assert torch.all(
            torch.eq(target, 0) + torch.eq(target, 1) + torch.eq(target, -1)
        ), "Target values should be either 0 OR 1 OR -1"

    def update(self, model_output, target):
        """
        Update the scores and targets
        """
        self.validate(model_output, target)
        self.verify_target(target)

        self._curr_sample_count += model_output.shape[0]

        curr_scores, curr_targets = self._scores, self._targets
        sample_count_so_far = curr_scores.shape[0]
        self._scores = torch.zeros(
            int(self._curr_sample_count[0]), self.num_classes, dtype=torch.float32
        )
        self._targets = torch.zeros(
            int(self._curr_sample_count[0]), self.num_classes, dtype=torch.int8
        )

        if sample_count_so_far > 0:
            self._scores[:sample_count_so_far, :] = curr_scores
            self._targets[:sample_count_so_far, :] = curr_targets
        self._scores[sample_count_so_far:, :] = model_output
        self._targets[sample_count_so_far:, :] = target
        del curr_scores, curr_targets

    def validate(self, model_output, target):
        """
        Validate that the input to meter is valid
        """
        assert len(model_output.shape) == 2, "model_output should be a 2D tensor"
        assert len(target.shape) == 2, "target should be a 2D tensor"
        assert (
            model_output.shape[0] == target.shape[0]
        ), "Expect same shape in model output and target"
        assert (
            model_output.shape[1] == target.shape[1]
        ), "Expect same shape in model output and target"
        num_classes = target.shape[1]
        assert num_classes == self.num_classes, "number of classes is not consistent"
