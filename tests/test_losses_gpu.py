# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
import torch.multiprocessing as mp
from vissl.losses.barlow_twins_loss import BarlowTwinsCriterion
from vissl.losses.simclr_info_nce_loss import SimclrInfoNCECriterion
from vissl.utils.test_utils import gpu_test, init_distributed_on_file, with_temp_files


class TestSimClrCriterionOnGpu(unittest.TestCase):
    """
    Specific tests on SimCLR going further than just doing a forward pass
    """

    @staticmethod
    def worker_fn(gpu_id: int, world_size: int, batch_size: int, sync_file: str):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        embeddings = torch.full(
            size=(batch_size, 3), fill_value=float(gpu_id), requires_grad=True
        ).cuda(gpu_id)
        gathered = SimclrInfoNCECriterion.gather_embeddings(embeddings)
        if world_size == 1:
            assert gathered.equal(
                torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=f"cuda:{gpu_id}"
                )
            )
        if world_size == 2:
            assert gathered.equal(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    device=f"cuda:{gpu_id}",
                )
            )
        assert gathered.requires_grad

    @gpu_test(gpu_count=1)
    def test_gather_embeddings_word_size_1(self) -> None:
        with with_temp_files(count=1) as sync_file:
            WORLD_SIZE = 1
            BATCH_SIZE = 2
            mp.spawn(
                self.worker_fn,
                args=(WORLD_SIZE, BATCH_SIZE, sync_file),
                nprocs=WORLD_SIZE,
            )

    @gpu_test(gpu_count=2)
    def test_gather_embeddings_word_size_2(self) -> None:
        with with_temp_files(count=1) as sync_file:
            WORLD_SIZE = 2
            BATCH_SIZE = 2
            mp.spawn(
                self.worker_fn,
                args=(WORLD_SIZE, BATCH_SIZE, sync_file),
                nprocs=WORLD_SIZE,
            )


class TestBarlowTwinsCriterionOnGpu(unittest.TestCase):
    """
    Specific tests on Barlow Twins going further than just doing a forward pass
    """

    @staticmethod
    def worker_fn(gpu_id: int, world_size: int, batch_size: int, sync_file: str):
        init_distributed_on_file(
            world_size=world_size, gpu_id=gpu_id, sync_file=sync_file
        )
        EMBEDDING_DIM = 128
        criterion = BarlowTwinsCriterion(
            lambda_=0.0051, scale_loss=0.024, embedding_dim=EMBEDDING_DIM
        )
        embeddings = torch.randn(
            (batch_size, EMBEDDING_DIM), dtype=torch.float32, requires_grad=True
        ).cuda()
        criterion(embeddings).backward()

    @gpu_test(gpu_count=1)
    def test_backward_world_size_1(self) -> None:
        with with_temp_files(count=1) as sync_file:
            WORLD_SIZE = 1
            BATCH_SIZE = 2
            mp.spawn(
                self.worker_fn,
                args=(WORLD_SIZE, BATCH_SIZE, sync_file),
                nprocs=WORLD_SIZE,
            )

    @gpu_test(gpu_count=2)
    def test_backward_world_size_2(self) -> None:
        with with_temp_files(count=1) as sync_file:
            WORLD_SIZE = 2
            BATCH_SIZE = 2
            mp.spawn(
                self.worker_fn,
                args=(WORLD_SIZE, BATCH_SIZE, sync_file),
                nprocs=WORLD_SIZE,
            )
