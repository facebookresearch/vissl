# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import List, Union

import torch
import torch.nn.functional as F
from classy_vision.generic.util import is_on_gpu
from classy_vision.losses import ClassyLoss, register_loss
from torch import Tensor, nn
from vissl.utils.hydra_config import AttrDict


class SmoothCrossEntropy(torch.nn.modules.CrossEntropyLoss):
    """
    Cross entropy loss that can accommodate smoothed labels
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) > 1:
            log_probs = F.log_softmax(input, 1)
            # TODO: Implement weight and ignore_index
            return -torch.mean(torch.sum(log_probs * target, dim=1))
        else:
            return F.cross_entropy(
                input, target, weight=self.weight, ignore_index=self.ignore_index
            )


@register_loss("cross_entropy_multiple_output_single_target")
class CrossEntropyMultipleOutputSingleTargetLoss(ClassyLoss):
    """
    Intializer for the sum cross-entropy loss. For a single
    tensor, this is equivalent to the cross-entropy loss. For a
    list of tensors, this computes the sum of the cross-entropy
    losses for each tensor in the list against the target. Can accommodate
    target vectors, e.g. smoothed labels.

    Config params:
        weight: weight of sample, optional
        ignore_index: sample should be ignored for loss, optional
        reduction: specifies reduction to apply to the output, optional
        temperature: specify temperature for softmax. Default 1.0
    """

    def __init__(self, loss_config: AttrDict):
        super(CrossEntropyMultipleOutputSingleTargetLoss, self).__init__()
        self._weight = None
        self._ignore_index = -1
        self._losses = torch.nn.modules.ModuleList([])
        self._normalize_output = False
        self._temperature = loss_config["temperature"]
        if "weight" in loss_config:
            self._weight = loss_config["weight"]
        if "ignore_index" in loss_config:
            self._ignore_index = loss_config["ignore_index"]
        if "normalize_output" in loss_config:
            self._normalize_output = loss_config["normalize_output"]

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates CrossEntropyMultipleOutputSingleTargetLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            CrossEntropyMultipleOutputSingleTargetLoss instance.
        """
        return cls(loss_config)

    def _create_loss_function(self):
        copy_to_gpu = is_on_gpu(self._losses)
        logging.info(
            "Instantiating "
            "CrossEntropyMultipleOutputSingleTargetLoss, which"
            "internally uses SmoothCrossEntropy loss to accommodate"
            "label smoothing, but defaults to vanilla cross-entropy "
            "if provided single-target labels."
        )
        self._losses.append(
            SmoothCrossEntropy(weight=self._weight, ignore_index=self._ignore_index)
        )
        if copy_to_gpu:
            self._losses.cuda()
        return self

    def forward(
        self, output: Union[torch.Tensor, List[torch.Tensor]], target: torch.Tensor
    ):
        """
        For each output and single target, loss is calculated.
        The returned loss value is the sum loss across all outputs.
        """
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

            assert (
                target.max().item() < pred.shape[1]
            ), f"pred.shape[1]={pred.shape[1]} and target.max().item()={target.max().item()}"
            if idx >= len(self._losses):
                self._create_loss_function()
            loss += self._losses[idx](normalized_pred / self._temperature, target)
        return loss
