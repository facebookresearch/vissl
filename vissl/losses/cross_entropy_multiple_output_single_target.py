# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from classy_vision.generic.util import is_on_gpu
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn, Tensor
from vissl.config import AttrDict


@dataclass
class EnsembleOutput:
    outputs: torch.Tensor  # Shape ensemble_size, batch_size, pred_size

    def cpu(self):
        return EnsembleOutput(self.outputs.cpu())


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


class CrossEntropyMultipleOutputSingleTargetCriterion(nn.Module):
    """
    Sum cross entropy loss:
    - For a single tensor, this is equivalent to the cross-entropy loss.
    - For a list of tensors, this computes the sum of the cross-entropy
    losses for each tensor in the list against the target.

    Can accommodate target vectors, e.g. smoothed labels.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        normalize_output: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        assert 0.0 <= label_smoothing <= 1.0

        self._weight = weight
        self._ignore_index = ignore_index
        self._losses = torch.nn.modules.ModuleList([])
        self._normalize_output = normalize_output
        self._temperature = temperature
        self._label_smoothing = label_smoothing

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
        for i, pred in enumerate(output):
            if i >= len(self._losses):
                self._losses.append(self._create_loss_function())

            if isinstance(pred, EnsembleOutput):
                # Shape batch_size, ensemble_size, pred_size -> batch_size * ensemble_size, pred_size
                ensemble_size, batch_size, pred_size = pred.outputs.shape
                pred = torch.reshape(
                    pred.outputs, (ensemble_size * batch_size, pred_size)
                )
                local_target = (
                    target.unsqueeze(0).expand((ensemble_size, batch_size)).flatten()
                )
            else:
                local_target = target

            assert (
                local_target.max().item() < pred.shape[1]
            ), f"pred.shape[1]={pred.shape[1]} and target.max().item()={target.max().item()}"

            if self._normalize_output:
                pred = nn.functional.normalize(pred, dim=1, p=2)

            if self._label_smoothing > 0.0:
                local_target = self.apply_label_smoothing(
                    local_target,
                    num_labels=pred.shape[1],
                    label_smoothing=self._label_smoothing,
                )

            loss += self._losses[i](pred / self._temperature, local_target)
        return loss

    def _create_loss_function(self):
        copy_to_gpu = is_on_gpu(self._losses)
        criterion = SmoothCrossEntropy(
            weight=self._weight, ignore_index=self._ignore_index
        )
        return criterion.cuda() if copy_to_gpu else criterion

    @staticmethod
    def apply_label_smoothing(
        target: torch.Tensor, num_labels: int, label_smoothing: float
    ):
        batch_size = target.shape[0]
        smoothed_targets = torch.full(
            size=(batch_size, num_labels),
            fill_value=label_smoothing / num_labels,
            device=target.device,
        )
        one_hot = torch.nn.functional.one_hot(target, num_classes=num_labels)
        smoothed_targets += (1 - label_smoothing) * one_hot
        return smoothed_targets


@register_loss("cross_entropy_multiple_output_single_target")
class CrossEntropyMultipleOutputSingleTargetLoss(ClassyLoss):
    """
    Initializer for the sum cross-entropy loss. For a single
    tensor, this is equivalent to the cross-entropy loss. For a
    list of tensors, this computes the sum of the cross-entropy
    losses for each tensor in the list against the target. Can accommodate
    target vectors, e.g. smoothed labels.

    Config params:
        weight: weight of sample, optional
        ignore_index: sample should be ignored for loss, optional
        reduction: specifies reduction to apply to the output, optional
        temperature: specify temperature for softmax. Default 1.0
        label_smoothing: specific a label smoothing factor between 0.0 and 1.0 (default is 0.0)
    """

    def __init__(self, loss_config: AttrDict):
        super(CrossEntropyMultipleOutputSingleTargetLoss, self).__init__()
        self._temperature = loss_config["temperature"]
        self._weight = loss_config.get("weight", None)
        self._ignore_index = loss_config.get("ignore_index", -1)
        self._normalize_output = loss_config.get("normalize_output", False)
        self._label_smoothing = loss_config.get("label_smoothing", 0.0)
        self.criterion = CrossEntropyMultipleOutputSingleTargetCriterion(
            weight=self._weight,
            temperature=self._temperature,
            ignore_index=self._ignore_index,
            normalize_output=self._normalize_output,
            label_smoothing=self._label_smoothing,
        )

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

    def forward(
        self, output: Union[torch.Tensor, List[torch.Tensor]], target: torch.Tensor
    ):
        """
        For each output and single target, loss is calculated.
        The returned loss value is the sum loss across all outputs.
        """
        return self.criterion(output, target)
