# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.models as models
from fairscale.nn import FullyShardedDataParallel
from vissl.utils.layer_memory_tracking import (
    find_best_reset_points,
    LayerwiseMemoryTracker,
    ProcessGroupTracker,
)
from vissl.utils.test_utils import (
    gpu_test,
    init_distributed_on_file,
    with_temp_files,
    with_timing,
)


class TestLayerMemoryTracking(unittest.TestCase):
    @gpu_test(gpu_count=1)
    def test_memory_tracking(self) -> None:

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

    @staticmethod
    def _layer_memory_tracking_worker(gpu_id: int, sync_file: str, world_size: int):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(gpu_id)

        batch_size = 16
        fake_inputs = torch.randn(size=(batch_size, 10)).cuda(gpu_id)
        fake_targets = torch.randn(size=(batch_size, 10)).cuda(gpu_id)
        fake_criterion = nn.MSELoss()

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Create a global group and a tracker around it
        group = dist.new_group()
        group = ProcessGroupTracker(group)

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 10).cuda(gpu_id),
            nn.ReLU(),
            FullyShardedDataParallel(
                nn.Linear(10, 10).cuda(gpu_id),
                flatten_parameters=False,
                process_group=group,
            ),
            nn.ReLU(),
            FullyShardedDataParallel(
                nn.Linear(10, 10).cuda(gpu_id),
                flatten_parameters=True,
                process_group=group,
            ),
        )
        model = model.cuda(gpu_id)
        model = FullyShardedDataParallel(
            model, flatten_parameters=False, process_group=group
        )

        # Setup the tracking of the model
        tracker = LayerwiseMemoryTracker()
        tracker.monitor(model)

        # Fake forward / backward pass
        fake_criterion(model(fake_inputs), fake_targets).backward()

        # Collect results of all gathers (the feature specific to FSDP)
        tracker.stop()
        all_gathered_traces = [
            (t.module_name, t.all_gathered, t.cumul_all_gathered)
            for t in tracker.memory_traces
            if t.all_gathered > 0
        ]
        assert all_gathered_traces == [
            ("_fsdp_wrapped_module._fpw_module.0", 440, 440),
            (
                "_fsdp_wrapped_module._fpw_module.2._fsdp_wrapped_module._fpw_module",
                440,
                880,
            ),
            (
                "_fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module._fpw_module",
                440,
                880,
            ),
            (
                "_fsdp_wrapped_module._fpw_module.4._fsdp_wrapped_module._fpw_module",
                440,
                0,
            ),
            (
                "_fsdp_wrapped_module._fpw_module.2._fsdp_wrapped_module._fpw_module",
                440,
                0,
            ),
        ], f"Expected {all_gathered_traces}"

    @gpu_test(gpu_count=2)
    def test_memory_tracking_fsdp(self) -> None:
        with with_temp_files(count=1) as sync_file:
            world_size = 2
            mp.spawn(
                self._layer_memory_tracking_worker,
                (sync_file, world_size),
                nprocs=world_size,
            )

    @gpu_test(gpu_count=1)
    def test_memory_tracking_performance_impact(self) -> None:
        torch.manual_seed(0)
        model = models.resnet18()
        with with_timing("no_tracking"):
            model(torch.randn(size=(1, 3, 224, 224)))
        with with_timing("with_tracking"):
            tracker = LayerwiseMemoryTracker()
            tracker.monitor(model)
            model(torch.randn(size=(1, 3, 224, 224)))

    def test_find_best_reset_points(self) -> None:
        """
        Verify that the reset points are correctly computed
        """
        activations = [10, 8, 8, 9, 7, 7, 5, 4, 4]

        # Check boundary condition: no checkpoints
        memory, split_points = find_best_reset_points(activations, nb_checkpoints=0)
        self.assertEqual(memory, sum(activations))

        # Check boundary condition: checkpoints everywhere
        memory, split_points = find_best_reset_points(
            activations, nb_checkpoints=len(activations)
        )
        self.assertEqual(memory, max(activations))

        # Check one checkpoint allocation
        memory, split_points = find_best_reset_points(activations, nb_checkpoints=1)
        self.assertEqual(memory, 35)
        self.assertEqual(split_points, [4])
        self.assertEqual(sum(activations[: split_points[0]]), 35)
        self.assertEqual(sum(activations[split_points[0] :]), 27)

        # Check multiple checkpoint allocation
        memory, split_points = find_best_reset_points(activations, nb_checkpoints=2)
        self.assertEqual(memory, 24)
        delimiters = [0] + split_points + [len(activations)]
        splits_memory = [
            sum(activations[i:j]) for i, j in zip(delimiters[:-1], delimiters[1:])
        ]
        self.assertEqual(max(splits_memory), memory)

    @gpu_test(gpu_count=1)
    def test_find_best_reset_points_performance(self) -> None:
        """
        Test that the algorithm is O(N**2) complexity for N activations
        """
        import numpy as np

        activations_1000 = list(np.random.randint(low=0, high=1_000_000, size=1_000))
        activations_2000 = list(np.random.randint(low=0, high=1_000_000, size=2_000))
        nb_checkpoints = 10
        with with_timing(name="best_reset_points_1000") as timer_1000:
            find_best_reset_points(activations_1000, nb_checkpoints=nb_checkpoints)
        with with_timing(name="best_reset_points_2000") as timer_2000:
            find_best_reset_points(activations_2000, nb_checkpoints=nb_checkpoints)
        self.assertGreaterEqual(timer_2000.elapsed_time_ms, timer_1000.elapsed_time_ms)
        self.assertLessEqual(timer_2000.elapsed_time_ms, timer_1000.elapsed_time_ms * 6)
