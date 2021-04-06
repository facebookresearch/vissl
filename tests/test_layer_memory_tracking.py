# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import contextlib
import unittest

import torch
import torch.nn as nn
import torchvision.models as models

from vissl.utils.layer_memory_tracking import LayerwiseMemoryTracker


class TestLayerMemoryTracking(unittest.TestCase):

    def test_memory_tracking(self):
        if not torch.cuda.is_available():
            return

        # Create a model with a hierarchy of modules
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ),
            nn.Flatten(start_dim=1),
            nn.Sequential(
                nn.Linear(64, 2),
                nn.ReLU(inplace=True),
            )
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
            tracker.forward_traces[2].memory_activations,
            "ReLU(inplace=False) should allocate activations"
        )
        self.assertEqual(
            0,
            tracker.forward_traces[6].memory_activations,
            "ReLU(inplace=True) should NOT allocate activations"
        )

        # Verify that overall memory tracking is sound
        summary = tracker.summary
        self.assertGreaterEqual(summary["total_forward_diff"], summary["total_forward_activation"])
        print(summary)

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

    def test_memory_tracking_performance_impact(self):
        torch.manual_seed(0)
        model = models.resnet18()
        with self.with_timing("no_tracking"):
            model(torch.randn(size=(1, 3, 224, 224)))
        with self.with_timing("with_tracking"):
            tracker = LayerwiseMemoryTracker()
            tracker.monitor(model)
            model(torch.randn(size=(1, 3, 224, 224)))
