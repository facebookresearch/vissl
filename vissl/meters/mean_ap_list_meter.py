# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pprint
from typing import List, Union

import torch
from classy_vision.generic.util import is_pos_int
from classy_vision.meters import ClassyMeter, register_meter
from vissl.config import AttrDict
from vissl.meters.mean_ap_meter import MeanAPMeter


@register_meter("mean_ap_list_meter")
class MeanAPListMeter(ClassyMeter):
    """
    Meter to calculate mean AP metric for multi-label image classification task
    on multiple output single target.

    Supports Single target and multiple output. A list of mean AP meters is
    constructed and each output has a meter associated.

    Args:
        meters_config (AttrDict): config containing the meter settings

    meters_config should specify the num_meters and meter_names
    """

    def __init__(self, meters_config: AttrDict):
        self.meters_config = meters_config
        num_meters = self.meters_config["num_meters"]
        meter_names = self.meters_config["meter_names"]
        assert is_pos_int(num_meters), "num_meters must be positive"
        self._num_meters = num_meters
        self._meters = [
            MeanAPMeter.from_config(meters_config) for _ in range(self._num_meters)
        ]
        self._meter_names = meter_names
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
        return "mean_ap_list_meter"

    @property
    def value(self):
        """
        Value of the meter globally synced. For each output, mean AP and AP for each class is
        returned.
        """
        val_dict = {}
        for ind, meter in enumerate(self._meters):
            meter_val = meter.value
            sample_count = meter._scores.shape[0]
            val_dict[ind] = {"val": meter_val, "sample_count": sample_count}
        output_dict = {}
        output_dict["mAP"] = {}
        output_dict["AP"] = {}
        for ind in range(len(self._meters)):
            meter_name = self._meter_names[ind] if (len(self._meter_names) > 0) else ind
            val = 100.0 * round(float(val_dict[ind]["val"]["mAP"]), 6)
            ap_matrix = val_dict[ind]["val"]["AP"].tolist()
            # we could have several meters with the same name. We append the result
            # to the dict.
            if meter_name not in output_dict["mAP"]:
                output_dict["mAP"][meter_name] = [val]
                output_dict["AP"][meter_name] = ap_matrix
            else:
                output_dict["mAP"][meter_name].append(val)
                output_dict["AP"][meter_name].append(ap_matrix)
        for k in output_dict["mAP"]:
            if len(output_dict["mAP"][k]) == 1:
                output_dict["mAP"][k] = output_dict["mAP"][k][0]
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
            meter_states[ind] = {"state": state}
        return meter_states

    def set_classy_state(self, state):
        """
        Set the state of each meter
        """
        assert len(state) == len(self._meters), "Incorrect state dict for meters"
        for ind, meter in enumerate(self._meters):
            meter.set_classy_state(state[ind]["state"])

    def __repr__(self):
        repr_dict = {
            "name": self.name,
            "num_meters": self._num_meters,
            "value": self.value,
        }
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
            probs = torch.nn.Sigmoid()(output)
            meter.update(probs, target)

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
