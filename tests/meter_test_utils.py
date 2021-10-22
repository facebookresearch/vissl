# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch


UPDATE_SIGNAL = 0
VALUE_SIGNAL = 1
SHUTDOWN_SIGNAL = 2
TIMEOUT = 100


class ClassificationMeterTest(unittest.TestCase):
    def _apply_updates_and_test_meter(
        self, meter, model_output, target, expected_value, **kwargs
    ):
        """
        Runs a valid meter test. Does not reset meter before / after running
        """
        if not isinstance(model_output, list):
            model_output = [model_output]

        if not isinstance(target, list):
            target = [target]

        for i in range(len(model_output)):
            meter.update(model_output[i], target[i], **kwargs)

        meter.sync_state()
        meter_value = meter.value
        for key, val in expected_value.items():
            val *= 100.0
            self.assertTrue(
                key in meter_value, msg="{0} not in meter value!".format(key)
            )
            if torch.is_tensor(meter_value[key][0]):
                self.assertTrue(
                    torch.all(torch.eq(meter_value[key][0], val)),
                    msg="{0} meter value mismatch!".format(key),
                )
            else:
                self.assertAlmostEqual(
                    meter_value[key][0],
                    val,
                    places=4,
                    msg="{0} meter value mismatch!".format(key),
                )

    def _validate_meter_inputs(self, meter, model_outputs, targets):
        for i in range(len(model_outputs)):
            meter.validate(model_outputs[i].size(), targets[i].size())

    def meter_update_and_reset_test(
        self, meter, model_outputs, targets, expected_value, **kwargs
    ):
        """
        This test verifies that a single update on the meter is successful,
        resets the meter, then applies the update again.
        """
        # If a single output is provided, wrap in list
        if not isinstance(model_outputs, list):
            model_outputs = [model_outputs]
            targets = [targets]

        self._validate_meter_inputs(meter, model_outputs, targets)

        self._apply_updates_and_test_meter(
            meter, model_outputs, targets, expected_value, **kwargs
        )

        meter.reset()

        # Verify reset works by reusing single update test
        self._apply_updates_and_test_meter(
            meter, model_outputs, targets, expected_value, **kwargs
        )

    def meter_get_set_classy_state_test(
        self, meters, model_outputs, targets, expected_value, **kwargs
    ):
        """
        Tests get and set classy state methods of meter.
        """
        assert len(meters) == 2, "Incorrect number of meters passed to test"
        assert (
            len(model_outputs) == 2
        ), "Incorrect number of model_outputs passed to test"
        assert len(targets) == 2, "Incorrect number of targets passed to test"
        meter0 = meters[0]
        meter1 = meters[1]

        meter0.update(model_outputs[0], targets[0], **kwargs)
        meter1.update(model_outputs[1], targets[1], **kwargs)

        meter0.sync_state()
        value0 = meter0.value

        meter1.sync_state()
        value1 = meter1.value
        for key, val in value0.items():
            val = val[0] * 100.0
            if torch.is_tensor(value1[key][0]):
                self.assertFalse(
                    torch.all(torch.eq(value1[key][0], val)),
                    msg="{0} meter values should not be same!".format(key),
                )
            else:
                self.assertNotEqual(
                    value1[key],
                    val,
                    msg="{0} meter values should not be same!".format(key),
                )

        meter0.set_classy_state(meter1.get_classy_state())
        value0 = meter0.value
        for key, val in value0.items():
            val = val[0]
            if torch.is_tensor(value1[key]):
                self.assertTrue(
                    torch.all(torch.eq(value1[key][0], val)),
                    msg="{0} meter value mismatch after state transfer!".format(key),
                )
                self.assertTrue(
                    torch.all(torch.eq(value1[key], expected_value[key])),
                    msg="{0} meter value mismatch from ground truth!".format(key),
                )
            else:
                self.assertAlmostEqual(
                    value1[key][0],
                    val,
                    places=4,
                    msg="{0} meter value mismatch after state transfer!".format(key),
                )
                self.assertAlmostEqual(
                    value1[key][0],
                    expected_value[key] * 100.0,
                    places=4,
                    msg="{0} meter value mismatch from ground truth!".format(key),
                )
