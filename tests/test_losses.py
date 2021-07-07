# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from collections import namedtuple

import torch
from classy_vision.generic.distributed_util import set_cpu_device
from parameterized import parameterized
from utils import ROOT_LOSS_CONFIGS, SSLHydraConfig
from vissl.losses.barlow_twins_loss import BarlowTwinsCriterion
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

    def test_barlow_twins_loss(self):
        loss_layer = BarlowTwinsCriterion(
            lambda_=0.0051, scale_loss=0.024, embedding_dim=EMBEDDING_DIM
        )
        _ = loss_layer(self._get_embedding())


class TestBarlowTwinsCriterion(unittest.TestCase):
    """
    Specific tests on Barlow Twins going further than just doing a forward pass
    """

    def test_barlow_twins_backward(self):
        EMBEDDING_DIM = 3
        criterion = BarlowTwinsCriterion(
            lambda_=0.0051, scale_loss=0.024, embedding_dim=EMBEDDING_DIM
        )
        embeddings = torch.randn((4, EMBEDDING_DIM), requires_grad=True)

        self.assertTrue(embeddings.grad is None)
        criterion(embeddings).backward()
        self.assertTrue(embeddings.grad is not None)
        with torch.no_grad():
            next_embeddings = embeddings - embeddings.grad  # gradient descent
            self.assertTrue(criterion(next_embeddings) < criterion(embeddings))


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
