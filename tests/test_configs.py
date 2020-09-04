# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import unittest

from hydra.errors import ConfigCompositionException, HydraException
from parameterized import parameterized
from utils import (
    BENCHMARK_CONFIGS,
    INTEGRATION_TEST_CONFIGS,
    PRETRAIN_CONFIGS,
    SSLHydraConfig,
)
from vissl.utils.hydra_config import convert_to_attrdict


logger = logging.getLogger("__name__")


class TestConfigsFail(unittest.TestCase):
    def test_cfg_fail_on_empty(self):
        try:
            SSLHydraConfig.from_configs()
            self.fail("We should fail if config is not specified")
        except ConfigCompositionException:
            # we must specify the base config otherwise it fails
            pass


class TestConfigsPass(unittest.TestCase):
    def test_load_cfg_success(self):
        # simply load from the config and this should pass
        self.assertTrue(
            SSLHydraConfig.from_configs(["config=test/integration_test/quick_simclr"]),
            "config must be loaded successfully",
        )


class TestConfigsComposition(unittest.TestCase):
    def test_cfg_composition(self):
        # compose the configs and check that the model is changed
        cfg = SSLHydraConfig.from_configs(
            [
                "config=test/integration_test/quick_simclr",
                "+config/pretrain/simclr/models=resnext101",
            ]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        self.assertEqual(
            config.MODEL.TRUNK.TRUNK_PARAMS.RESNETS.DEPTH,
            101,
            "config composition failed",
        )


class TestConfigsFailComposition(unittest.TestCase):
    def test_cfg_fail_composition(self):
        # compose the configs and check that the model is changed
        try:
            SSLHydraConfig.from_configs(
                [
                    "config=test/integration_test/quick_simclr",
                    "config/pretrain/simclr/models=resnext101",
                ]
            )
            self.fail(
                "We should fail for invalid composition. "
                "+ is necessary as the group does not exists in defaults"
            )
        except HydraException:
            pass


class TestConfigsCliComposition(unittest.TestCase):
    def test_cfg_cli_composition(self):
        # compose the configs and check that the model is changed
        cfg = SSLHydraConfig.from_configs(
            [
                "config=test/integration_test/quick_simclr",
                "+config/pretrain/simclr/models=resnext101",
                "config.MODEL.TRUNK.TRUNK_PARAMS.RESNETS.GROUPS=32",
                "config.MODEL.TRUNK.TRUNK_PARAMS.RESNETS.WIDTH_PER_GROUP=16",
            ]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        self.assertEqual(
            config.MODEL.TRUNK.TRUNK_PARAMS.RESNETS.GROUPS,
            32,
            "config composition failed",
        )
        self.assertEqual(
            config.MODEL.TRUNK.TRUNK_PARAMS.RESNETS.WIDTH_PER_GROUP,
            16,
            "config composition failed",
        )


class TestConfigsKeyAddition(unittest.TestCase):
    def test_cfg_key_addition(self):
        # compose the configs and check that the new key is inserted
        cfg = SSLHydraConfig.from_configs(
            [
                "config=test/integration_test/quick_simclr",
                "+config.LOSS.simclr_info_nce_loss.buffer_params.MY_TEST_KEY=dummy",
            ]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        self.assertTrue(
            "MY_TEST_KEY" in config.LOSS.simclr_info_nce_loss.buffer_params,
            "something went wrong, new key not added. Fail.",
        )


class TestBenchmarkConfigs(unittest.TestCase):
    @parameterized.expand(BENCHMARK_CONFIGS)
    def test_benchmark_config(self, filepath):
        logger.warning(f"Loading {filepath}")
        self.assertTrue(
            SSLHydraConfig.from_configs([filepath]),
            "config must be loaded successfully",
        )


class TestPretrainConfigs(unittest.TestCase):
    @parameterized.expand(PRETRAIN_CONFIGS)
    def test_pretrain_config(self, filepath):
        logger.warning(f"Loading {filepath}")
        self.assertTrue(
            SSLHydraConfig.from_configs([filepath]),
            "config must be loaded successfully",
        )


class TestIntegrationTestConfigs(unittest.TestCase):
    @parameterized.expand(INTEGRATION_TEST_CONFIGS)
    def test_integration_test_config(self, filepath):
        logger.warning(f"Loading {filepath}")
        self.assertTrue(
            SSLHydraConfig.from_configs([filepath]),
            "config must be loaded successfully",
        )
