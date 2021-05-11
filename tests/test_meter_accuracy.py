#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from vissl.meters.accuracy_list_meter import AccuracyListMeter
from tests.generic.meter_test_utils import ClassificationMeterTest

from classy_vision.meters import build_meter
import vissl.meters  # noqa: F401,F811


class TestAccuracyListMeter(ClassificationMeterTest):
    def test_accuracy_meter_registry(self):
        meter = build_meter({"name": "accuracy_list_meter", "num_meters": 1, "topk_values": [
                            1, 3], "meter_names": []})
        self.assertTrue(isinstance(meter, AccuracyListMeter))

    def test_single_meter_update_and_reset(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update.
        """
        meter = AccuracyListMeter(num_meters=1, topk_values=[
                                  1, 2], meter_names=[])

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_output = torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]])

        # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
        target = torch.tensor([0, 1, 2])

        # Only the first sample has top class correct, first and third
        # sample have correct class in top 2
        expected_value = {"top_1": 1 / 3.0, "top_2": 2 / 3.0}

        self.meter_update_and_reset_test(
            meter, model_output, target, expected_value)

    def test_double_meter_update_and_reset(self):
        meter = AccuracyListMeter(num_meters=1, topk_values=[
                                  1, 2], meter_names=[])

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score...two batches in this test
        model_outputs = [
            torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),
            torch.tensor([[3, 2, 1], [1, 3, 2], [1, 3, 2]]),
        ]

        # Class 0 is the correct class for sample 1, class 2 for
        # sample 2, etc, in both batches
        targets = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])]

        # First batch has top-1 accuracy of 1/3.0, top-2 accuracy of 2/3.0
        # Second batch has top-1 accuracy of 2/3.0, top-2 accuracy of 3/3.0
        expected_value = {"top_1": 3 / 6.0, "top_2": 5 / 6.0}

        self.meter_update_and_reset_test(
            meter, model_outputs, targets, expected_value)

    def test_single_meter_update_and_reset_onehot(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update with onehot target.
        """
        meter = AccuracyListMeter(num_meters=1, topk_values=[
                                  1, 2], meter_names=[])

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_output = torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]])

        # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
        target = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Only the first sample has top class correct, first and third
        # sample have correct class in top 2
        expected_value = {"top_1": 1 / 3.0, "top_2": 2 / 3.0}

        self.meter_update_and_reset_test(
            meter, model_output, target, expected_value)

    def test_single_meter_update_and_reset_multilabel(self):
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update with multilabel target.
        """
        meter = AccuracyListMeter(num_meters=1, topk_values=[
                                  1, 2], meter_names=[])

        # Batchsize = 7, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_output = torch.tensor(
            [
                [3, 2, 1],
                [3, 1, 2],
                [1, 3, 2],
                [1, 2, 3],
                [2, 1, 3],
                [2, 3, 1],
                [1, 3, 2],
            ]
        )

        target = torch.tensor(
            [
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [1, 0, 1],
            ]
        )

        # 1st, 4th, 5th, 6th sample has top class correct, 2nd and 7th have at least
        # one correct class in top 2.
        expected_value = {"top_1": 4 / 7.0, "top_2": 6 / 7.0}

        self.meter_update_and_reset_test(
            meter, model_output, target, expected_value)

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
        # Expected value is the expected value of meter1
        meters = [AccuracyListMeter(num_meters=1, topk_values=[1, 2], meter_names=[
        ]), AccuracyListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])]

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_outputs = [
            torch.tensor([[1, 2, 3], [1, 2, 3], [2, 3, 1]]),
            torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),
        ]

        # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
        targets = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])]

        # Value for second update
        expected_value = {"top_1": 1 / 3.0, "top_2": 2 / 3.0}

        self.meter_get_set_classy_state_test(
            meters, model_outputs, targets, expected_value
        )

    def test_meter_distributed(self):
        # Meter0 will execute on one process, Meter1 on the other
        meters = [AccuracyListMeter(num_meters=1, topk_values=[1, 2], meter_names=[
        ]), AccuracyListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])]

        # Batchsize = 3, num classes = 3, score is a value in {1, 2,
        # 3}...3 is the highest score
        model_outputs = [
            torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),  # Meter 0
            torch.tensor([[3, 2, 1], [1, 3, 2], [1, 3, 2]]),  # Meter 1
            torch.tensor([[3, 2, 1], [3, 1, 2], [1, 3, 2]]),  # Meter 0
            torch.tensor([[3, 2, 1], [1, 3, 2], [1, 3, 2]]),  # Meter 1
        ]

        # Class 0 is the correct class for sample 1, class 2 for sample 2, etc
        targets = [
            torch.tensor([0, 1, 2]),  # Meter 0
            torch.tensor([0, 1, 2]),  # Meter 1
            torch.tensor([0, 1, 2]),  # Meter 0
            torch.tensor([0, 1, 2]),  # Meter 1
        ]

        # In first two updates there are 3 correct top-2, 5 correct in top 2
        # The same occurs in the second two updates and is added to first
        expected_values = [
            # After one update to each meter
            {"top_1": 3 / 6.0, "top_2": 5 / 6.0},
            # After two updates to each meter
            {"top_1": 6 / 12.0, "top_2": 10 / 12.0},
        ]

        self.meter_distributed_test(
            meters, model_outputs, targets, expected_values)
