#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing
import queue
import tempfile
import unittest

import torch


UPDATE_SIGNAL = 0
VALUE_SIGNAL = 1
SHUTDOWN_SIGNAL = 2
TIMEOUT = 100


def _get_value_or_raise_error(qout, qerr):
    try:
        return qout.get(timeout=TIMEOUT)
    except queue.Empty:
        raise qerr.get(timeout=TIMEOUT)


def _run(qin, qout, qerr, func, *args):
    try:
        func(qin, qout, *args)
    except Exception as e:
        print(e)
        qerr.put(e)


def _meter_worker(qin, qout, meter, is_train, world_size, rank, filename):
    backend = "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        init_method="file://{filename}".format(filename=filename),
        world_size=world_size,
        rank=rank,
    )
    # Listen for commands on queues
    while True:
        try:
            signal, val = qin.get()
        except queue.Empty:
            continue

        if signal == UPDATE_SIGNAL:
            meter.update(val[0], val[1])

        elif signal == VALUE_SIGNAL:
            meter.sync_state()
            qout.put(meter.value)

        elif signal == SHUTDOWN_SIGNAL:
            break

        else:
            raise NotImplementedError("Bad signal value")

    return


class ClassificationMeterTest(unittest.TestCase):
    def setUp(self):
        self.mp = multiprocessing.get_context("spawn")
        self.processes = []

    def tearDown(self):
        for p in self.processes:
            p.terminate()

    def _spawn(self, func, *args):
        name = "process #%d" % len(self.processes)
        qin = self.mp.Queue()
        qout = self.mp.Queue()
        qerr = self.mp.Queue()
        qio = (qin, qout, qerr)
        args = qio + (func,) + args
        process = self.mp.Process(target=_run, name=name, args=args, daemon=True)
        process.start()
        self.processes.append(process)
        return qio

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
            val *=100.0 
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

    def _values_match_expected_value(self, value0, value1, expected_value):
        for key, val in expected_value.items():
            val*=100.0
            self.assertTrue(key in value0, msg="{0} not in meter value!".format(key))
            self.assertTrue(key in value1, msg="{0} not in meter value!".format(key))
            if torch.is_tensor(val):
                self.assertTrue(
                    torch.all(torch.eq(value0[key][0], val)),
                    "{0} meter value mismatch!".format(key),
                )
                self.assertTrue(
                    torch.all(torch.eq(value1[key][0], val)),
                    "{0} meter value mismatch!".format(key),
                )
            else:
                self.assertAlmostEqual(
                    value0[key][0],
                    val,
                    places=4,
                    msg="{0} meter value mismatch!".format(key),
                )
                self.assertAlmostEqual(
                    value1[key][0],
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

    def meter_invalid_meter_input_test(self, meter, model_output, target):
        # Invalid model
        with self.assertRaises(AssertionError):
            meter.validate(model_output.shape, target.shape)

    def meter_invalid_update_test(self, meter, model_output, target, **kwargs):
        """
        Runs a valid meter test. Does not reset meter before / after running
        """
        if not isinstance(model_output, list):
            model_output = [model_output]

        if not isinstance(target, list):
            target = [target]

        with self.assertRaises(AssertionError):
            for i in range(len(model_output)):
                meter.update(model_output[i], target[i], **kwargs)

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
            val = val[0]*100.0
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
                    expected_value[key]*100.0,
                    places=4,
                    msg="{0} meter value mismatch from ground truth!".format(key),
                )

    def _spawn_all_meter_workers(self, world_size, meters, is_train):
        filename = tempfile.NamedTemporaryFile(delete=True).name
        qins = []
        qerrs = []
        qouts = []

        for i in range(world_size):
            qin, qout, qerr = self._spawn(
                _meter_worker, meters[i], is_train, world_size, i, filename
            )
            qins.append(qin)
            qouts.append(qout)
            qerrs.append(qerr)

        return qins, qouts, qerrs

    def meter_distributed_test(
        self, meters, model_outputs, targets, expected_values, is_train=False
    ):
        """
        Sets up two processes each with a given meter on that process.
        Verifies that sync code path works.
        """
        world_size = len(meters)
        assert world_size == 2, "This test only works for world_size of 2"
        assert len(model_outputs) == 4, (
            "Test assumes 4 model outputs, "
            "0, 2 passed to meter0 and 1, 3 passed to meter1"
        )
        assert (
            len(targets) == 4
        ), "Test assumes 4 targets, 0, 2 passed to meter0 and 1, 3 passed to meter1"
        assert len(expected_values) == 2, (
            "Test assumes 2 expected values, "
            "first is result of applying updates 0,1 to the meter, "
            "second is result of applying all 4 updates to meter"
        )

        qins, qouts, qerrs = self._spawn_all_meter_workers(
            world_size, meters, is_train=is_train
        )

        # First update each meter, then get value from each meter
        qins[0].put_nowait((UPDATE_SIGNAL, (model_outputs[0], targets[0])))
        qins[1].put_nowait((UPDATE_SIGNAL, (model_outputs[1], targets[1])))

        qins[0].put_nowait((VALUE_SIGNAL, None))
        qins[1].put_nowait((VALUE_SIGNAL, None))

        value0 = _get_value_or_raise_error(qouts[0], qerrs[0])
        value1 = _get_value_or_raise_error(qouts[1], qerrs[1])
        self._values_match_expected_value(value0, value1, expected_values[0])

        # Verify that calling value again does not break things
        qins[0].put_nowait((VALUE_SIGNAL, None))
        qins[1].put_nowait((VALUE_SIGNAL, None))

        value0 = _get_value_or_raise_error(qouts[0], qerrs[0])
        value1 = _get_value_or_raise_error(qouts[1], qerrs[1])
        self._values_match_expected_value(value0, value1, expected_values[0])

        # Second, update each meter, then get value from each meter
        qins[0].put_nowait((UPDATE_SIGNAL, (model_outputs[2], targets[2])))
        qins[1].put_nowait((UPDATE_SIGNAL, (model_outputs[3], targets[3])))

        qins[0].put_nowait((VALUE_SIGNAL, None))
        qins[1].put_nowait((VALUE_SIGNAL, None))

        value0 = _get_value_or_raise_error(qouts[0], qerrs[0])
        value1 = _get_value_or_raise_error(qouts[1], qerrs[1])
        self._values_match_expected_value(value0, value1, expected_values[1])

        qins[0].put_nowait((SHUTDOWN_SIGNAL, None))
        qins[1].put_nowait((SHUTDOWN_SIGNAL, None))
