# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Union

import torch
from classy_vision.generic.util import is_on_gpu
from classy_vision.losses import ClassyLoss, SoftTargetCrossEntropyLoss, register_loss
from torch import nn
from vissl.utils.hydra_config import AttrDict


@register_loss("soft_target_cross_entropy_multiple_output_single_target")
class SoftTargetCrossEntropyMultipleOutputSingleTargetLoss(ClassyLoss):
    def __init__(self, loss_config: AttrDict):
        """
        Intializer for the sum cross-entropy loss. For a single
        tensor, this is equivalent to the cross-entropy loss. For a
        list of tensors, this computes the sum of the cross-entropy
        losses for each tensor in the list against the target.

        Config params:
            "weight": weight of sample, [NOT IMPLEMENTED]
            "ignore_index": sample should be ignored for loss, optional
            "reduction": specifies reduction to apply to the output, optional
            "normalize_targets": by default, we normalize based on the count
            "normalize_output": Whether to L2 normalize the outputs
        """
        super(SoftTargetCrossEntropyMultipleOutputSingleTargetLoss, self).__init__()
        self.loss_config = loss_config
        self._losses = torch.nn.modules.ModuleList([])
        self._normalize_output = loss_config.get("normalize_output", False)

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        return cls(loss_config)

    def _create_loss_function(self):
        copy_to_gpu = is_on_gpu(self._losses)
        self._losses.append(SoftTargetCrossEntropyLoss.from_config(self.loss_config))
        if copy_to_gpu:
            self._losses.cuda()
        return self

    def forward(
        self, output: Union[torch.Tensor, List[torch.Tensor]], target: torch.Tensor
    ):
        if isinstance(output, torch.Tensor):
            output = [output]
        assert isinstance(
            output, list
        ), "Model output should be a list of tensors. Got Type {}".format(type(output))
        assert torch.is_tensor(target), "Target should be a tensor. Got Type {}".format(
            type(target)
        )
        loss = 0
        for idx, pred in enumerate(output):
            normalized_pred = pred
            if self._normalize_output:
                normalized_pred = nn.functional.normalize(pred, dim=1, p=2)

            if idx >= len(self._losses):
                self._create_loss_function()
            loss += self._losses[idx](normalized_pred, target)
        return loss
