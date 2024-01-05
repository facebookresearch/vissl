# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from fairscale.nn import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from vissl.optimizers.larc_fsdp import LARC_FSDP
from vissl.utils.test_utils import gpu_test, init_distributed_on_file, with_temp_files


class TestLarcFSDP(unittest.TestCase):
    """
    LARC requires a distributed norm computation when the parameters
    and gradients are sharded.

    This test ensures that the implementation of LARC_FSDP does
    this computation correctly.
    """

    @staticmethod
    def _norm_computation_worker(gpu_id: int, sync_file: str, world_size: int):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True

        num_iterations = 10
        batch_size = 128
        torch.manual_seed(gpu_id)
        fake_inputs = torch.randn(size=(num_iterations, batch_size, 129))
        fake_targets = torch.randn(size=(num_iterations, batch_size))

        losses = {}
        for with_fsdp in [False, True]:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            losses[with_fsdp] = []

            # Create a simple model
            model = nn.Sequential(nn.Linear(129, 128), nn.ReLU(), nn.Linear(128, 10))
            model = model.cuda(gpu_id)

            # Setting up FSDP vs DDP with LARC
            larc_config = {"clip": False, "trust_coefficient": 0.01, "eps": 0.00000001}
            optimizer = optim.SGD(
                model.parameters(), lr=1e-2, weight_decay=1e-4, momentum=0.9
            )
            if with_fsdp:
                model = FullyShardedDataParallel(model, flatten_parameters=False)
                optimizer = LARC_FSDP(optimizer, distributed_norm=True, **larc_config)
            else:
                model = DistributedDataParallel(model, device_ids=[gpu_id])
                optimizer = LARC_FSDP(optimizer, distributed_norm=False, **larc_config)

            # Training loop
            criterion = nn.MSELoss()
            for iteration in range(num_iterations):
                fake_input = fake_inputs[iteration].cuda(gpu_id)
                fake_target = fake_targets[iteration].cuda(gpu_id)
                output = model(fake_input)
                loss = criterion(output.sum(axis=-1), fake_target)
                if gpu_id == 0:
                    losses[with_fsdp].append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if gpu_id == 0:
            for with_fsdp in [False, True]:
                print(losses[with_fsdp])
                if world_size > 1:
                    losses[with_fsdp] = [round(loss, 5) for loss in losses[with_fsdp]]
            assert losses[False] == losses[True]

    @gpu_test(gpu_count=1)
    def test_local_norm_computations(self) -> None:
        """
        Sanity check: the sharded and non-sharded norms should be trivially
        the same when the number of GPU involved in 1 (no sharding)
        """
        with with_temp_files(count=1) as sync_file:
            world_size = 1
            mp.spawn(
                self._norm_computation_worker,
                (sync_file, world_size),
                nprocs=world_size,
            )

    @gpu_test(gpu_count=2)
    def test_norm_computations(self) -> None:
        """
        Trying with 2 GPUs: the sharded computation should span across GPUs
        and lead to sensibly the same results as normal DDP with LARC
        """
        with with_temp_files(count=1) as sync_file:
            world_size = 2
            mp.spawn(
                self._norm_computation_worker,
                (sync_file, world_size),
                nprocs=world_size,
            )
