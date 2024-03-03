# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint
from typing import List, Union

import torch
from classy_vision.generic.util import is_pos_int
from classy_vision.meters import ClassyMeter, PrecisionAtKMeter, register_meter
from vissl.config import AttrDict


@register_meter("precision_at_k_list_meter")
class PrecisionAtKListMeter(ClassyMeter):
    """
    Meter to calculate precision@k.

    Supports multi-target and multiple output. A list of precision meters is
    constructed and each output has a meter associated. Note that Precision@k is
    different than vanilla Precision.

    Example:
        target = [0 1 0 1 0]  # Correct classes are 1, 3
        pred = [0.06, 0.41, 0.04, 0.39, 0.1]  # Top-1 prediction is 1, top-3 is 1, 3, 0

        Precision@1: 1 correct / 1 predicted = 1.0
        Precision@3: 2 correct / 3 predicted = 0.666

    Args:
        num_meters: number of meters and hence we have same number of outputs
        topk_values: list of int `k` values. Example: [1, 5]
        meter_names: list of str indicating the name of meter. Usually corresponds
                     to the output layer name.
    """

    def __init__(self, num_meters: int, topk_values: List[int], meter_names: List[str]):
        super().__init__()

        assert is_pos_int(num_meters), "num_meters must be positive"
        assert isinstance(topk_values, list), "topk_values must be a list"
        assert len(topk_values) > 0, "topk_values list should have at least one element"
        assert [
            is_pos_int(x) for x in topk_values
        ], "each value in topk_values must be >= 1"
        self._num_meters = num_meters
        self._topk_values = topk_values
        self._meters = [
            PrecisionAtKMeter(self._topk_values) for _ in range(self._num_meters)
        ]
        self._meter_names = meter_names
        self.reset()

    @classmethod
    def from_config(cls, meters_config: AttrDict):
        """
        Get the PrecisionAtKListMeter instance from the user defined config
        """
        return cls(
            num_meters=meters_config["num_meters"],
            topk_values=meters_config["topk_values"],
            meter_names=meters_config["meter_names"],
        )

    @property
    def name(self):
        """
        Name of the meter
        """
        return "precision_at_k_list_meter"

    @property
    def value(self):
        """
        Value of the meter globally synced. For each output, all the top-k values are
        returned. If there are several meters attached to the same layer
        name, a list of top-k values will be returned for that layer name meter.
        """
        val_dict = {}
        for ind, meter in enumerate(self._meters):
            meter_val = meter.value
            sample_count = meter._total_sample_count
            val_dict[ind] = {}
            val_dict[ind]["val"] = meter_val
            val_dict[ind]["sample_count"] = sample_count

        # also create dict w.r.t top-k
        output_dict = {}
        for k in self._topk_values:
            top_k_str = f"top_{k}"
            output_dict[top_k_str] = {}
            for ind in range(len(self._meters)):
                meter_name = (
                    self._meter_names[ind] if (len(self._meter_names) > 0) else ind
                )
                val = 100.0 * round(float(val_dict[ind]["val"][top_k_str]), 6)
                # we could have several meters with the same name. We append the result
                # to the dict.
                if meter_name not in output_dict[top_k_str]:
                    output_dict[top_k_str][meter_name] = [val]
                else:
                    output_dict[top_k_str][meter_name].append(val)
        for topk in output_dict:
            for k in output_dict[topk]:
                if len(output_dict[topk][k]) == 1:
                    output_dict[topk][k] = output_dict[topk][k][0]
        return output_dict

    def sync_state(self):
        """
        Globally syncing the state of each meter across all the trainers.
        """
        for _, meter in enumerate(self._meters):
            meter.sync_state()

    def get_classy_state(self):
        """
        Returns the states of each meter
        """
        meter_states = {}
        for ind, meter in enumerate(self._meters):
            state = meter.get_classy_state()
            meter_states[ind] = {}
            meter_states[ind]["state"] = state
        return meter_states

    def set_classy_state(self, state):
        """
        Set the state of each meter
        """
        assert len(state) == len(self._meters), "Incorrect state dict for meters"
        for ind, meter in enumerate(self._meters):
            meter.set_classy_state(state[ind]["state"])

    def __repr__(self):
        value = self.value
        # convert top_k list into csv format for easy copy pasting
        for k in self._topk_values:
            top_k_str = f"top_{k}"
            hr_format = ["%.1f" % (100 * x) for x in value[top_k_str]]
            value[top_k_str] = ",".join(hr_format)

        repr_dict = {"name": self.name, "num_meters": self._num_meters, "value": value}
        return pprint.pformat(repr_dict, indent=2)

    def update(
        self,
        model_output: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
    ):
        """
        Updates the value of the meter for the given model output list and targets.

        Args:
            model_output: list of tensors of shape (B, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B).

        NOTE: For binary classification, C=2.
        """
        if isinstance(model_output, torch.Tensor):
            model_output = [model_output]
        assert isinstance(model_output, list)
        assert len(model_output) == self._num_meters
        for meter, output in zip(self._meters, model_output):
            meter.update(output, target)

    def reset(self):
        """
        Reset all the meters
        """
        [x.reset() for x in self._meters]

    def validate(self, model_output_shape, target_shape):
        """
        Not implemented
        """
        pass
