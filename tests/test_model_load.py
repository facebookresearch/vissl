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
from vissl.utils.hydra_config import convert_to_attrdict, hydra_compose
from vissl.utils.test_utils import parameterized_random


logger = logging.getLogger("__name__")


def is_fsdp_model_config(config) -> bool:
    """
    Exclude FSDP configurations from the test model load
    as FSDP models requires:
    - multiple GPU to be instantiated
    - lots of GPU memory to be instantiated
    """
    return "fsdp" in config.TRAINER.TASK_NAME or "fsdp" in config.MODEL.TRUNK.NAME


def is_huge_trunk(config) -> bool:
    if config.MODEL.TRUNK.NAME == "resnet_sk":
        return True
    if config.MODEL.TRUNK.NAME == "regnet":
        if config.MODEL.TRUNK.REGNET.get("name", "") == "regnet_y_256gf":
            return True
        if config.MODEL.TRUNK.REGNET.get("depth", 0) >= 27:
            return True
    if config.MODEL.TRUNK.NAME == "vision_transformer":
        if config.MODEL.TRUNK.VISION_TRANSFORMERS.HIDDEN_DIM > 1024:
            return True
    return False


def is_big_model_too_big_for_ci(config) -> bool:
    return is_fsdp_model_config(config) or is_huge_trunk(config)


class TestBenchmarkModel(unittest.TestCase):
    @parameterized_random(
        BENCHMARK_MODEL_CONFIGS, ratio=0.25, keep=lambda config: "models" not in config
    )
    def test_benchmark_model(self, filepath: str):
        cfg = hydra_compose([filepath, "config.DISTRIBUTED.NUM_PROC_PER_NODE=1"])
        _, config = convert_to_attrdict(cfg)
        if not is_big_model_too_big_for_ci(config):
            logger.warning(f"Creating model: {filepath}")
            build_model(config.MODEL, config.OPTIMIZER)
        else:
            logger.warning(f"Ignoring model: {filepath}")
        return True


class TestPretrainModel(unittest.TestCase):
    @parameterized.expand(PRETRAIN_MODEL_CONFIGS)
    def test_pretrain_model(self, filepath: str):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs([filepath])
        _, config = convert_to_attrdict(cfg.default_cfg)
        if not is_big_model_too_big_for_ci(config):
            build_model(config.MODEL, config.OPTIMIZER)


class TestIntegrationTestModel(unittest.TestCase):
    @parameterized.expand(INTEGRATION_TEST_CONFIGS)
    def test_integration_test_model(self, filepath: str):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs([filepath])
        _, config = convert_to_attrdict(cfg.default_cfg)
        if not is_big_model_too_big_for_ci(config):
            build_model(config.MODEL, config.OPTIMIZER)
