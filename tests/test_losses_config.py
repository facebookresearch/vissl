# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from collections import namedtuple

from classy_vision.generic.distributed_util import set_cpu_device
from utils import ROOT_LOSS_CONFIGS, SSLHydraConfig
from vissl.trainer.train_task import SelfSupervisionTask
from vissl.utils.hydra_config import convert_to_attrdict, hydra_compose
from vissl.utils.test_utils import parameterized_parallel


logger = logging.getLogger("__name__")

set_cpu_device()

BATCH_SIZE = 2048
EMBEDDING_DIM = 128
NUM_CROPS = 2
BUFFER_PARAMS_STRUCT = namedtuple(
    "BUFFER_PARAMS_STRUCT", ["effective_batch_size", "world_size", "embedding_dim"]
)
BUFFER_PARAMS = BUFFER_PARAMS_STRUCT(BATCH_SIZE, 1, EMBEDDING_DIM)


class TestRootConfigsLossesBuild(unittest.TestCase):
    @parameterized_parallel(ROOT_LOSS_CONFIGS)
    def test_loss_build(self, filepath):
        logger.warning(f"Loading {filepath}")
        cfg = hydra_compose(
            [
                filepath,
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
            ]
        )
        _, config = convert_to_attrdict(cfg)
        task = SelfSupervisionTask.from_config(config)
        task.datasets, _ = task.build_datasets()
        try:
            self.assertTrue(task._build_loss(), f"failed to build loss: {filepath}")
        except Exception as e:
            logger.error(f"failed to build loss: {filepath}")
            raise e
        return True

    def test_pytorch_loss(self) -> None:
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
