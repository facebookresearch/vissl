# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest

import torch
import torch.nn as nn
import torchvision.models as models
from vissl.utils.layer_memory_tracking import LayerwiseMemoryTracker


class TestLayerMemoryTracking(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires at least 1 GPU")
    def test_memory_tracking(self):

        # Create a model with a hierarchy of modules
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ),
            nn.Flatten(start_dim=1),
            nn.Sequential(nn.Linear(64, 2), nn.ReLU(inplace=True)),
        ).cuda()

        # Track a fake forward / backward
        tracker = LayerwiseMemoryTracker()
        tracker.monitor(model)
        x = torch.randn(size=(2, 3, 224, 224)).cuda()
        target = torch.LongTensor([0, 1]).cuda()
        criterion = nn.CrossEntropyLoss()
        criterion(model(x), target).backward()

        # Verify that only leaf modules are tracked
        tracked_names = {trace.module_name for trace in tracker.memory_traces}
        expected_names = {"0.0", "0.1", "0.2", "0.3", "1", "2.0", "2.1"}
        self.assertEqual(expected_names, tracked_names)

        # Verify that memory tracking for ReLU is sound
        self.assertEqual(
            25233408,
            tracker.forward_traces[2].event.memory_activations,
            "ReLU(inplace=False) should allocate activations",
        )
        self.assertEqual(
            0,
            tracker.forward_traces[6].event.memory_activations,
            "ReLU(inplace=True) should NOT allocate activations",
        )

        # Verify that overall memory tracking is sound
        summary = tracker.summary
        self.assertGreaterEqual(
            summary.total_forward_allocations, summary.total_activation_allocations
        )

        top_act_producers = summary.top_forward_activation_producers[:3]
        self.assertEqual("0.0", top_act_producers[0].module_name)
        self.assertEqual("0.1", top_act_producers[1].module_name)
        self.assertEqual("0.2", top_act_producers[2].module_name)
        self.assertEqual(7168, top_act_producers[0].module_params)
        self.assertEqual(512, top_act_producers[1].module_params)
        self.assertEqual(0, top_act_producers[2].module_params)
        for trace in top_act_producers:
            self.assertEqual(25233408, trace.event.memory_activations)

    @contextlib.contextmanager
    def with_timing(self, name: str):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        yield
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(name, ":", elapsed_time_ms, "ms")

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires at least 1 GPU")
    def test_memory_tracking_performance_impact(self):
        torch.manual_seed(0)
        model = models.resnet18()
        with self.with_timing("no_tracking"):
            model(torch.randn(size=(1, 3, 224, 224)))
        with self.with_timing("with_tracking"):
            tracker = LayerwiseMemoryTracker()
            tracker.monitor(model)
            model(torch.randn(size=(1, 3, 224, 224)))
