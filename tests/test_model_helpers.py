# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from parameterized import param, parameterized
from torch.nn.parallel import DistributedDataParallel
from vissl.config import AttrDict
from vissl.models.model_helpers import convert_sync_bn, split_world_in_process_groups
from vissl.utils.env import set_env_vars
from vissl.utils.test_utils import gpu_test, init_distributed_on_file, with_temp_files


class TestModelHelpers(unittest.TestCase):
    def test_split_in_process_groups(self) -> None:
        # Standards use cases
        pids = split_world_in_process_groups(world_size=9, group_size=3)
        self.assertEqual(pids, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        pids = split_world_in_process_groups(world_size=9, group_size=4)
        self.assertEqual(pids, [[0, 1, 2, 3], [4, 5, 6, 7], [8]])

        # Pathological use cases
        pids = split_world_in_process_groups(world_size=0, group_size=1)
        self.assertEqual(pids, [])
        pids = split_world_in_process_groups(world_size=5, group_size=6)
        self.assertEqual(pids, [[0, 1, 2, 3, 4]])

    @parameterized.expand([param(group_size=1), param(group_size=2)])
    @gpu_test(gpu_count=2)
    def test_synch_bn_pytorch(self, group_size: int):
        world_size = 2
        with with_temp_files(count=1) as sync_file:
            mp.spawn(
                self._test_synch_bn_pytorch_worker,
                (world_size, group_size, sync_file),
                nprocs=world_size,
            )

    @gpu_test(gpu_count=4)
    def test_synch_bn_pytorch_large_world(self) -> None:
        world_size = 4
        group_size = 2
        with with_temp_files(count=1) as sync_file:
            mp.spawn(
                self._test_synch_bn_pytorch_worker,
                (world_size, group_size, sync_file),
                nprocs=world_size,
            )

    @staticmethod
    def _test_synch_bn_pytorch_worker(
        gpu_id: int, world_size: int, group_size: int, sync_file: str
    ):
        torch.cuda.set_device(gpu_id)
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )

        config = AttrDict(
            {
                "MODEL": {
                    "SYNC_BN_CONFIG": {
                        "SYNC_BN_TYPE": "pytorch",
                        "GROUP_SIZE": group_size,
                    }
                },
                "DISTRIBUTED": {
                    "NUM_PROC_PER_NODE": world_size,
                    "NUM_NODES": 1,
                    "NCCL_DEBUG": False,
                    "NCCL_SOCKET_NTHREADS": 4,
                },
            }
        )
        set_env_vars(local_rank=gpu_id, node_id=0, cfg=config)

        channels = 8
        model = nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        model = convert_sync_bn(config, model).cuda(gpu_id)
        model = DistributedDataParallel(model, device_ids=[gpu_id])
        x = torch.full(size=(5, channels, 4, 4), fill_value=float(gpu_id))
        model(x)
        running_mean = model.module[0].running_mean.cpu()
        print(gpu_id, running_mean)
        if group_size == 1:
            if gpu_id == 0:
                assert torch.allclose(
                    running_mean, torch.full(size=(8,), fill_value=0.0)
                )
            elif gpu_id == 1:
                assert torch.allclose(
                    running_mean, torch.full(size=(8,), fill_value=0.1)
                )
        else:
            if gpu_id in {0, 1}:
                assert torch.allclose(
                    running_mean, torch.full(size=(8,), fill_value=0.05)
                )
