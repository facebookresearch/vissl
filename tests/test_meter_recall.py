#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from vissl.meters.recall_list_meter import RecallListMeter
from tests.generic.meter_test_utils import ClassificationMeterTest

from classy_vision.meters import build_meter
import vissl.meters as meters


class TestRecallListMeter(ClassificationMeterTest):
    def test_recall_meter_registry(self):
        meter = build_meter({'name': 'recall_list_meter', 'num_meters': 1, 'topk_values': [1, 3], 'meter_names': []})
        self.assertTrue(isinstance(meter, RecallListMeter))

    def test_single_meter_update_and_reset(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update.
        """
        meter = RecallListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])

        # Batchsize = 3, num classes = 3, score is probability of class
        model_output = torch.tensor(
            [
                [0.2, 0.4, 0.4],  # top-1: 1/2, top-2: 1/2
                [0.2, 0.65, 0.15],  # top-1: 1, top-2: 1/0
                [0.33, 0.33, 0.34],  # top-1: 2, top-2: 2/0?1
            ]
        )

        # One-hot encoding, 1 = positive for class
        # sample-1: 1, sample-2: 0, sample-3: 0,1,2
        target = torch.tensor([[0, 1, 0], [1, 0, 0], [1, 1, 1]])

        # Note for ties, we select randomly, so we should not use ambiguous ties
        expected_value = {"top_1": 2 / 5.0, "top_2": 4 / 5.0}

        self.meter_update_and_reset_test(meter, model_output, target, expected_value)

    def test_double_meter_update_and_reset(self):
        meter = RecallListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])

        # Batchsize = 3, num classes = 3, score is probability of class
        model_outputs = [
            torch.tensor([[0.3, 0.4, 0.3], [0.2, 0.65, 0.15], [0.33, 0.33, 0.34]]),
            torch.tensor([[0.05, 0.4, 0.05], [0.15, 0.65, 0.2], [0.4, 0.2, 0.4]]),
        ]

        # One-hot encoding, 1 = positive for class
        # batch-1: sample-1: 1, sample-2: 0, sample-3: 0,1,2
        # batch-2: sample-1: 1, sample-2: 1, sample-3: 1
        targets = [
            torch.tensor([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),
            torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        ]

        # First batch has top-1 recall of 2/5.0, top-2 recall of 4/5.0
        # Second batch has top-1 recall of 2/3.0, top-2 recall of 2/3.0
        expected_value = {"top_1": 4 / 8.0, "top_2": 6 / 8.0}

        self.meter_update_and_reset_test(meter, model_outputs, targets, expected_value)
    
    def test_meter_invalid_model_output(self):
        # TODO fill when meters execute validate
        pass

    def test_meter_invalid_target(self):
        # TODO fill when meters execute validate
        pass

    def test_meter_invalid_topk(self):
        # TODO fill when meters execute validate
        pass

    def test_meter_get_set_classy_state_test(self):
        # In this test we update meter0 with model_output0 & target0
        # and we update meter1 with model_output1 & target1 then
        # transfer the state from meter1 to meter0 and validate they
        # give same expected value.
        #
        # Expected value is the expected value of meter1 For this test
        # to work, top-1 / top-2 values of meter0 / meter1 should be
        # different
        meters = [RecallListMeter(num_meters=1, topk_values=[1, 2], meter_names=[]), RecallListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])]
        model_outputs = [
            torch.tensor([[0.05, 0.4, 0.05], [0.2, 0.65, 0.15], [0.33, 0.33, 0.34]]),
            torch.tensor([[0.05, 0.4, 0.05], [0.15, 0.65, 0.2], [0.4, 0.2, 0.4]]),
        ]
        targets = [
            torch.tensor([[0, 1, 0], [1, 0, 0], [1, 1, 0]]),
            torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        ]

        # Second update's expected value
        expected_value = {"top_1": 2 / 3.0, "top_2": 2 / 3.0}

        self.meter_get_set_classy_state_test(
            meters, model_outputs, targets, expected_value
        )

    def test_meter_distributed(self):
        # Meter0 will execute on one process, Meter1 on the other
        meters = [RecallListMeter(num_meters=1, topk_values=[1, 2], meter_names=[]), RecallListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])]

        # Batchsize = 3, num classes = 3, score is probability of class
        model_outputs = [
            torch.tensor(
                [[0.3, 0.4, 0.3], [0.2, 0.65, 0.15], [0.33, 0.33, 0.34]]
            ),  # Meter 0
            torch.tensor(
                [[0.05, 0.4, 0.05], [0.15, 0.65, 0.2], [0.4, 0.2, 0.4]]
            ),  # Meter 1
            torch.tensor(
                [[0.3, 0.4, 0.3], [0.2, 0.65, 0.15], [0.33, 0.33, 0.34]]
            ),  # Meter 0
            torch.tensor(
                [[0.05, 0.4, 0.05], [0.15, 0.65, 0.2], [0.4, 0.2, 0.4]]
            ),  # Meter 1
        ]

        # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
        targets = [
            torch.tensor([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),  # Meter 0
            torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),  # Meter 1
            torch.tensor([[0, 1, 0], [1, 0, 0], [1, 1, 1]]),  # Meter 0
            torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),  # Meter 1
        ]

        # In first two updates there are 4 correct top-1 out of 8
        # total, 6 correct in top 2 out of 8.  The same occurs in the
        # second two updates and is added to first
        expected_values = [
            {"top_1": 4 / 8.0, "top_2": 6 / 8.0},  # After one update to each meter
            {"top_1": 8 / 16.0, "top_2": 12 / 16.0},  # After two updates to each meter
        ]

        self.meter_distributed_test(meters, model_outputs, targets, expected_values)

    def test_non_onehot_target(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update.
        """
        meter = RecallListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])

        # Batchsize = 2, num classes = 3, score is probability of class
        model_outputs = [
            torch.tensor([[0.05, 0.4, 0.05], [0.15, 0.65, 0.2], [0.4, 0.2, 0.4]]),
            torch.tensor([[0.2, 0.4, 0.4], [0.2, 0.65, 0.15], [0.1, 0.8, 0.1]]),
        ]

        # One-hot encoding, 1 = positive for class
        targets = [
            torch.tensor([[1], [1], [1]]),  # [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
            torch.tensor([[0], [1], [2]]),  # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ]

        # Note for ties, we select randomly, so we should not use ambiguous ties
        # First batch has top-1 recall of 2/3.0, top-2 recall of 2/6.0
        # Second batch has top-1 recall of 1/3.0, top-2 recall of 4/6.0
        expected_value = {"top_1": 3 / 6.0, "top_2": 6 / 12.0}

        self.meter_update_and_reset_test(meter, model_outputs, targets, expected_value)

    def test_non_onehot_target_one_dim_target(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update with one dimensional targets.
        """
        meter = RecallListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])

        # Batchsize = 2, num classes = 3, score is probability of class
        model_outputs = [
            torch.tensor([[0.05, 0.4, 0.05], [0.15, 0.65, 0.2], [0.4, 0.2, 0.4]]),
            torch.tensor([[0.2, 0.4, 0.4], [0.2, 0.65, 0.15], [0.1, 0.8, 0.1]]),
        ]

        # One-hot encoding, 1 = positive for class
        targets = [
            torch.tensor([1, 1, 1]),  # [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
            torch.tensor([0, 1, 2]),  # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ]

        # Note for ties, we select randomly, so we should not use ambiguous ties
        # First batch has top-1 recall of 2/3.0, top-2 recall of 2/6.0
        # Second batch has top-1 recall of 1/3.0, top-2 recall of 4/6.0
        expected_value = {"top_1": 3 / 6.0, "top_2": 6 / 12.0}

        self.meter_update_and_reset_test(meter, model_outputs, targets, expected_value)
