#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pprint

from classy_vision.generic.util import is_pos_int
from classy_vision.meters import AccuracyMeter, ClassyMeter, register_meter


@register_meter("accuracy_list")
class AccuracyListMeter(ClassyMeter):
    """Meter to calculate top-k accuracy for single label
       image classification task.
    """

    def __init__(self, num_list, topk):
        """
        args:
            num_list: num outputs
            topk: list of int `k` values.
        """
        assert is_pos_int(num_list), "num_list must be positive"
        assert isinstance(topk, list), "topk must be a list"
        assert len(topk) > 0, "topk list should have at least one element"
        assert [is_pos_int(x) for x in topk], "each value in topk must be >= 1"
        self._num_list = num_list
        self._topk = topk
        self._meters = [AccuracyMeter(self._topk) for _ in range(self._num_list)]
        self.reset()

    @classmethod
    def from_config(cls, config):
        return cls(num_list=config["num_list"], topk=config["topk"])

    @property
    def name(self):
        return "accuracy_list"

    @property
    def value(self):
        val_dict = {}
        for ind, meter in enumerate(self._meters):
            meter_val = meter.value
            sample_count = meter._total_sample_count
            val_dict[ind] = {}
            val_dict[ind]["val"] = meter_val
            val_dict[ind]["sample_count"] = sample_count
        # also create dict wrt top-k
        output_dict = {}
        for k in self._topk:
            top_k_str = f"top_{k}"
            output_dict[top_k_str] = {}
            for ind in range(len(self._meters)):
                output_dict[top_k_str][ind] = 100.0 * round(
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
        for k in self._topk:
            top_k_str = f"top_{k}"
            hr_format = ["%.1f" % (100 * x) for x in value[top_k_str]]
            value[top_k_str] = ",".join(hr_format)

        repr_dict = {"name": self.name, "num_list": self._num_list, "value": value}
        return pprint.pformat(repr_dict, indent=2)

    def update(self, model_output, target):
        """
        args:
            model_output: list of tensors of shape (B, C) where each value is
                          either logit or class probability.
            target:       tensor of shape (B).
            Note: For binary classification, C=2.
        """
        assert isinstance(model_output, list)
        assert len(model_output) == self._num_list
        for (meter, output) in zip(self._meters, model_output):
            meter.update(output, target)

    def reset(self):
        [x.reset() for x in self._meters]

    def validate(self, model_output_shape, target_shape):
        pass
