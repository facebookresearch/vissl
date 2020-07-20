# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import unittest
from collections import namedtuple

import torch
from classy_vision.generic.distributed_util import set_cpu_device
from parameterized import parameterized
from utils import ROOT_OSS_CONFIGS, SSLHydraConfig
from vissl.ssl_losses.multicrop_simclr_info_nce_loss import (
    MultiCropSimclrInfoNCECriterion,
)
from vissl.ssl_losses.simclr_info_nce_loss import SimclrInfoNCECriterion
from vissl.ssl_losses.swav_loss import SwAVCriterion
from vissl.ssl_tasks import build_task
from vissl.utils.hydra_config import convert_to_attrdict


logger = logging.getLogger("__name__")


set_cpu_device()

BATCH_SIZE = 2048
EMBEDDING_DIM = 128
NMB_CROPS = 2
BUFFER_PARAMS_STRUCT = namedtuple(
    "BUFFER_PARAMS_STRUCT", ["EFFECTIVE_BATCH_SIZE", "WORLD_SIZE", "EMBEDDING_DIM"]
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
            buffer_params=BUFFER_PARAMS, temperature=0.1, nmb_crops=NMB_CROPS
        )
        embedding = torch.ones([BATCH_SIZE * NMB_CROPS, EMBEDDING_DIM])
        _ = loss_layer(embedding)

    def test_swav_loss(self):
        loss_layer = SwAVCriterion(
            temperature=0.1,
            crops_for_assign=[0, 1],
            nmb_crops=2,
            nmb_iters=3,
            epsilon=0.05,
            use_double_prec=False,
            nmb_prototypes=[3000],
            local_queue_length=0,
            embedding_dim=EMBEDDING_DIM,
        )
        _ = loss_layer(scores=self._get_embedding(), head_id=0)


class TestRootConfigsLossesBuild(unittest.TestCase):
    @parameterized.expand(ROOT_OSS_CONFIGS)
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
        task = build_task(config)
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
        task = build_task(config)
        task.datasets, _ = task.build_datasets()
        self.assertTrue(task._build_loss(), "failed to build loss")
