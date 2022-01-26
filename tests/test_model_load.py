# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from parameterized import parameterized
from utils import (
    BENCHMARK_MODEL_CONFIGS,
    INTEGRATION_TEST_CONFIGS,
    PRETRAIN_MODEL_CONFIGS,
    SSLHydraConfig,
)
from vissl.models import build_model
from vissl.utils.hydra_config import convert_to_attrdict


logger = logging.getLogger("__name__")


def is_fsdp_model_config(config) -> bool:
    """
    Exclude FSDP configurations from the test model load
    as FSDP models requires:
    - multiple GPU to be instantiated
    - lots of GPU memory to be instantiated
    """
    return "fsdp" in config.TRAINER.TASK_NAME or "fsdp" in config.MODEL.TRUNK.NAME


class TestBenchmarkModel(unittest.TestCase):
    @parameterized.expand(BENCHMARK_MODEL_CONFIGS)
    def test_benchmark_model(self, filepath: str):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs(
            [filepath, "config.DISTRIBUTED.NUM_PROC_PER_NODE=1"]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        if not is_fsdp_model_config(config):
            build_model(config.MODEL, config.OPTIMIZER)


class TestPretrainModel(unittest.TestCase):
    @parameterized.expand(PRETRAIN_MODEL_CONFIGS)
    def test_pretrain_model(self, filepath: str):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs([filepath])
        _, config = convert_to_attrdict(cfg.default_cfg)
        if not is_fsdp_model_config(config):
            build_model(config.MODEL, config.OPTIMIZER)


class TestIntegrationTestModel(unittest.TestCase):
    @parameterized.expand(INTEGRATION_TEST_CONFIGS)
    def test_integration_test_model(self, filepath: str):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs([filepath])
        _, config = convert_to_attrdict(cfg.default_cfg)
        if not is_fsdp_model_config(config):
            build_model(config.MODEL, config.OPTIMIZER)
