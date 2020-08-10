# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pprint
from typing import List, Union

import torch
from classy_vision.generic.util import is_pos_int
from classy_vision.meters import AccuracyMeter, ClassyMeter, register_meter
from vissl.utils.hydra_config import AttrDict


@register_meter("accuracy_list_meter")
class AccuracyListMeter(ClassyMeter):
    """
    Meter to calculate top-k accuracy for single label image classification task.
    """

    def __init__(self, num_meters: int, topk_values: List[int], meter_names: List[str]):
        """
        args:
            num_meters: number of meters and hence we have same number of outputs
            topk_values: list of int `k` values.
        """
        assert is_pos_int(num_meters), "num_meters must be positive"
        assert isinstance(topk_values, list), "topk_values must be a list"
        assert len(topk_values) > 0, "topk_values list should have at least one element"
        assert [
            is_pos_int(x) for x in topk_values
        ], "each value in topk_values must be >= 1"
        self._num_meters = num_meters
        self._topk_values = topk_values
        self._meters = [
            AccuracyMeter(self._topk_values) for _ in range(self._num_meters)
        ]
        self._meter_names = meter_names
        self.reset()

    @classmethod
    def from_config(cls, meters_config: AttrDict):
        return cls(
            num_meters=meters_config["num_meters"],
            topk_values=meters_config["topk_values"],
            meter_names=meters_config["meter_names"],
        )

    @property
    def name(self):
        return "accuracy_list_meter"

    @property
    def value(self):
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
                output_dict[top_k_str][meter_name] = 100.0 * round(
                    float(val_dict[ind]["val"][top_k_str]), 6
                )
        return output_dict

    def sync_state(self):
        for _, meter in enumerate(self._meters):
            meter.sync_state()

    def get_classy_state(self):
        """
        Contains the states of the meter
        """
        meter_states = {}
        for ind, meter in enumerate(self._meters):
            state = meter.get_classy_state()
            meter_states[ind] = {}
            meter_states[ind]["state"] = state
        return meter_states

    def set_classy_state(self, state):
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
        args:
            model_output: list of tensors of shape (B, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B).
            Note: For binary classification, C=2.
        """
        if isinstance(model_output, torch.Tensor):
            model_output = [model_output]
        assert isinstance(model_output, list)
        assert len(model_output) == self._num_meters
        for (meter, output) in zip(self._meters, model_output):
            meter.update(output, target)

    def reset(self):
        [x.reset() for x in self._meters]

    def validate(self, model_output_shape, target_shape):
        pass
