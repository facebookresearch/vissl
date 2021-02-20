# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import unittest
from collections import namedtuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from classy_vision.generic.distributed_util import set_cpu_device
from parameterized import parameterized
from utils import ROOT_LOSS_CONFIGS, SSLHydraConfig
from vissl.losses.multicrop_simclr_info_nce_loss import MultiCropSimclrInfoNCECriterion
from vissl.losses.simclr_info_nce_loss import SimclrInfoNCECriterion
from vissl.losses.swav_loss import SwAVCriterion
from vissl.trainer.train_task import SelfSupervisionTask
from vissl.utils.hydra_config import convert_to_attrdict


logger = logging.getLogger("__name__")

set_cpu_device()

BATCH_SIZE = 2048
EMBEDDING_DIM = 128
NUM_CROPS = 2
BUFFER_PARAMS_STRUCT = namedtuple(
    "BUFFER_PARAMS_STRUCT", ["effective_batch_size", "world_size", "embedding_dim"]
)
BUFFER_PARAMS = BUFFER_PARAMS_STRUCT(BATCH_SIZE, 1, EMBEDDING_DIM)


class TestLossesForward(unittest.TestCase):
    """
    Minimal testing of the losses: ensure that a forward pass with believable
    dimensions succeeds. This does not make them correct per say.
    """

    @staticmethod
    def _get_embedding():
        return torch.ones([BATCH_SIZE, EMBEDDING_DIM])

    def test_simclr_info_nce_loss(self):
        loss_layer = SimclrInfoNCECriterion(
            buffer_params=BUFFER_PARAMS, temperature=0.1
        )
        _ = loss_layer(self._get_embedding())

    def test_multicrop_simclr_info_nce_loss(self):
        loss_layer = MultiCropSimclrInfoNCECriterion(
            buffer_params=BUFFER_PARAMS, temperature=0.1, num_crops=NUM_CROPS
        )
        embedding = torch.ones([BATCH_SIZE * NUM_CROPS, EMBEDDING_DIM])
        _ = loss_layer(embedding)

    def test_swav_loss(self):
        loss_layer = SwAVCriterion(
            temperature=0.1,
            crops_for_assign=[0, 1],
            num_crops=2,
            num_iters=3,
            epsilon=0.05,
            use_double_prec=False,
            num_prototypes=[3000],
            local_queue_length=0,
            embedding_dim=EMBEDDING_DIM,
            temp_hard_assignment_iters=0,
            output_dir="",
        )
        _ = loss_layer(scores=self._get_embedding(), head_id=0)


class TestSimClrCriterion(unittest.TestCase):
    """
    Specific tests on SimCLR going further than just doing a forward pass
    """

    def test_simclr_info_nce_masks(self):
        BATCH_SIZE = 4
        WORLD_SIZE = 2
        buffer_params = BUFFER_PARAMS_STRUCT(
            BATCH_SIZE * WORLD_SIZE, WORLD_SIZE, EMBEDDING_DIM
        )
        criterion = SimclrInfoNCECriterion(buffer_params=buffer_params, temperature=0.1)
        self.assertTrue(
            criterion.pos_mask.equal(
                torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
        )
        self.assertTrue(
            criterion.neg_mask.equal(
                torch.tensor(
                    [
                        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                )
            )
        )

    def test_simclr_backward(self):
        EMBEDDING_DIM = 3
        BATCH_SIZE = 4
        WORLD_SIZE = 1
        buffer_params = BUFFER_PARAMS_STRUCT(
            BATCH_SIZE * WORLD_SIZE, WORLD_SIZE, EMBEDDING_DIM
        )
        criterion = SimclrInfoNCECriterion(buffer_params=buffer_params, temperature=0.1)
        embeddings = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            requires_grad=True,
        )

        self.assertTrue(embeddings.grad is None)
        criterion(embeddings).backward()
        self.assertTrue(embeddings.grad is not None)
        print(embeddings.grad)
        with torch.no_grad():
            next_embeddings = embeddings - embeddings.grad  # gradient descent
            self.assertTrue(criterion(next_embeddings) < criterion(embeddings))

    @staticmethod
    def worker_fn(gpu_id: int, world_size: int, batch_size: int):
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://0.0.0.0:1234",
            world_size=world_size,
            rank=gpu_id,
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

    def test_gather_embeddings_word_size_1(self):
        if torch.cuda.device_count() >= 1:
            WORLD_SIZE = 1
            BATCH_SIZE = 2
            mp.spawn(self.worker_fn, args=(WORLD_SIZE, BATCH_SIZE), nprocs=WORLD_SIZE)

    def test_gather_embeddings_word_size_2(self):
        if torch.cuda.device_count() >= 2:
            WORLD_SIZE = 2
            BATCH_SIZE = 2
            mp.spawn(self.worker_fn, args=(WORLD_SIZE, BATCH_SIZE), nprocs=WORLD_SIZE)


class TestRootConfigsLossesBuild(unittest.TestCase):
    @parameterized.expand(ROOT_LOSS_CONFIGS)
    def test_loss_build(self, filepath):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs(
            [
                filepath,
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
            ]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        task = SelfSupervisionTask.from_config(config)
        task.datasets, _ = task.build_datasets()
        self.assertTrue(task._build_loss(), "failed to build loss")

    def test_pytorch_loss(self):
        cfg = SSLHydraConfig.from_configs(
            [
                "config=test/integration_test/quick_simclr",
                "config.LOSS.name=CosineEmbeddingLoss",
                "+config.LOSS.CosineEmbeddingLoss.margin=1.0",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
            ]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        task = SelfSupervisionTask.from_config(config)
        task.datasets, _ = task.build_datasets()
        self.assertTrue(task._build_loss(), "failed to build loss")
