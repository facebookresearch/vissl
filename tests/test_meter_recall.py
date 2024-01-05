# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import vissl.meters  # noqa: F401,F811
from classy_vision.meters import build_meter
from meter_test_utils import ClassificationMeterTest
from vissl.meters.recall_at_k_list_meter import RecallAtKListMeter


class TestRecallAtKListMeter(ClassificationMeterTest):
    def test_recall_meter_registry(self) -> None:
        meter = build_meter(
            {
                "name": "recall_at_k_list_meter",
                "num_meters": 1,
                "topk_values": [1, 3],
                "meter_names": [],
            }
        )
        self.assertTrue(isinstance(meter, RecallAtKListMeter))

    def test_single_meter_update_and_reset(self) -> None:
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update.
        """
        meter = RecallAtKListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])

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

    def test_double_meter_update_and_reset(self) -> None:
        meter = RecallAtKListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])

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

    def test_meter_get_set_classy_state_test(self) -> None:
        # In this test we update meter0 with model_output0 & target0
        # and we update meter1 with model_output1 & target1 then
        # transfer the state from meter1 to meter0 and validate they
        # give same expected value.
        #
        # Expected value is the expected value of meter1 For this test
        # to work, top-1 / top-2 values of meter0 / meter1 should be
        # different
        meters = [
            RecallAtKListMeter(num_meters=1, topk_values=[1, 2], meter_names=[]),
            RecallAtKListMeter(num_meters=1, topk_values=[1, 2], meter_names=[]),
        ]
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

    def test_non_onehot_target(self) -> None:
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update.
        """
        meter = RecallAtKListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])

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

    def test_non_onehot_target_one_dim_target(self) -> None:
        """
        This test verifies that the meter works as expected on a single
        update + reset + same single update with one dimensional targets.
        """
        meter = RecallAtKListMeter(num_meters=1, topk_values=[1, 2], meter_names=[])

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
