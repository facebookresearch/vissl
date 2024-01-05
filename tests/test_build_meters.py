# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import vissl.meters  # noqa: F401,F811
from classy_vision.meters import build_meter


class TestBuildMeters(unittest.TestCase):
    def test_build_meters(self) -> None:
        configs = [
            {
                "name": "accuracy_list_meter",
                "num_meters": 1,
                "topk_values": [1, 3],
                "meter_names": [],
            },
            {
                "name": "precision_at_k_list_meter",
                "num_meters": 1,
                "topk_values": [1, 3],
                "meter_names": [],
            },
            {
                "name": "recall_at_k_list_meter",
                "num_meters": 1,
                "topk_values": [1, 3],
                "meter_names": [],
            },
        ]
        for config in configs:
            build_meter(config)

    def test_multi_update(self) -> None:
        meters = []
        configs = [
            {
                "name": "accuracy_list_meter",
                "num_meters": 1,
                "topk_values": [1, 3],
                "meter_names": [],
            },
            {
                "name": "precision_at_k_list_meter",
                "num_meters": 1,
                "topk_values": [1, 3],
                "meter_names": [],
            },
            {
                "name": "recall_at_k_list_meter",
                "num_meters": 1,
                "topk_values": [1, 3],
                "meter_names": [],
            },
        ]
        for config in configs:
            meters.append(build_meter(config))

        # One-hot encoding, 1 = positive for class
        # sample-1: 1, sample-2: 0, sample-3: 0,1,2
        target = torch.tensor([[0, 1, 0], [1, 0, 0], [1, 1, 1]])
        prev_values = {}

        for _ in range(50):

            # Batchsize = 3, num classes = 3, score is probability of class
            model_output = torch.rand((3, 3)).softmax(dim=1).cpu()

            for i, meter in enumerate(meters):
                if i in prev_values.keys():
                    assert str(meter.value) == prev_values[i]
                meter.update(model_output, target.cpu())
                prev_values[i] = str(meter.value)
