#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unittest

from classy_vision.meters import build_meter
import vissl.meters  # noqa: F401,F811


class TestBuildMeters(unittest.TestCase):
    def test_build_meters(self):
        meter_accuracy = build_meter({"name": "accuracy_list_meter", "num_meters": 1, "topk_values": [
                            1, 3], "meter_names": []})
        meter_precision = build_meter({'name': 'precision_list_meter',
                            'num_meters': 1, 'topk_values': [1, 3], 'meter_names': []})
        meter_recall = build_meter({'name': 'recall_list_meter', 'num_meters': 1, 'topk_values': [
                            1, 3], 'meter_names': []})
    
    def test_multi_update(self):
        meter_accuracy = build_meter({"name": "accuracy_list_meter", "num_meters": 1, "topk_values": [
                            1, 3], "meter_names": []})
        meter_precision = build_meter({'name': 'precision_list_meter',
                            'num_meters': 1, 'topk_values': [1, 3], 'meter_names': []})
        meter_recall = build_meter({'name': 'recall_list_meter', 'num_meters': 1, 'topk_values': [
                            1, 3], 'meter_names': []})
        meters = [meter_accuracy, meter_precision, meter_recall]
        

        # One-hot encoding, 1 = positive for class
        # sample-1: 1, sample-2: 0, sample-3: 0,1,2
        target = torch.tensor([[0, 1, 0], [1, 0, 0], [1, 1, 1]])
        prev_values = {}

        for _ in range(50):

            # Batchsize = 3, num classes = 3, score is probability of class
            model_output = torch.rand((3,3)).softmax(dim=1).cpu()

            for i,meter in enumerate(meters): 
                if i in prev_values.keys():
                    assert str(meter.value) == prev_values[i]
                meter.update(model_output, target.cpu()) 
                prev_values[i] = str(meter.value)
