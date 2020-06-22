#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import unittest

from parameterized import parameterized
from utils import (
    BENCHMARK_CONFIGS,
    INTEGRATION_TEST_CONFIGS,
    PRETRAIN_CONFIGS,
    SSLHydraConfig,
)
from vissl.models import build_model
from vissl.utils.hydra_config import convert_to_attrdict


logger = logging.getLogger("__name__")


class TestBenchmarkModel(unittest.TestCase):
    @parameterized.expand(BENCHMARK_CONFIGS)
    def test_benchmark_model(self, filepath):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs([filepath])
        _, config = convert_to_attrdict(cfg.default_cfg)
        build_model(config.MODEL, config.OPTIMIZER)


class TestPretrainModel(unittest.TestCase):
    @parameterized.expand(PRETRAIN_CONFIGS)
    def test_pretrain_model(self, filepath):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs([filepath])
        _, config = convert_to_attrdict(cfg.default_cfg)
        build_model(config.MODEL, config.OPTIMIZER)


class TestIntegrationTestModel(unittest.TestCase):
    @parameterized.expand(INTEGRATION_TEST_CONFIGS)
    def test_integration_test_model(self, filepath):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs([filepath])
        _, config = convert_to_attrdict(cfg.default_cfg)
        build_model(config.MODEL, config.OPTIMIZER)
