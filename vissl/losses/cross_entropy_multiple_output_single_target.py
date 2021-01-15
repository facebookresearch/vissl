# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Union

import torch
from classy_vision.generic.util import is_on_gpu
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.utils.hydra_config import AttrDict


@register_loss("cross_entropy_multiple_output_single_target")
class CrossEntropyMultipleOutputSingleTargetLoss(ClassyLoss):
    """
    Intializer for the sum cross-entropy loss. For a single
    tensor, this is equivalent to the cross-entropy loss. For a
    list of tensors, this computes the sum of the cross-entropy
    losses for each tensor in the list against the target.

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
        self._losses.append(
            torch.nn.modules.CrossEntropyLoss(
                weight=self._weight, ignore_index=self._ignore_index
            )
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
