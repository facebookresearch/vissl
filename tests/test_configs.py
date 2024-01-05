# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from hydra.errors import ConfigCompositionException, HydraException
from utils import (
    BENCHMARK_CONFIGS,
    INTEGRATION_TEST_CONFIGS,
    PRETRAIN_CONFIGS,
    SSLHydraConfig,
)
from vissl.utils.hydra_config import convert_to_attrdict, hydra_compose
from vissl.utils.test_utils import parameterized_parallel


logger = logging.getLogger("__name__")


class TestConfigsFail(unittest.TestCase):
    def test_cfg_fail_on_empty(self) -> None:
        try:
            SSLHydraConfig.from_configs()
            self.fail("We should fail if config is not specified")
        except ConfigCompositionException:
            # we must specify the base config otherwise it fails
            pass


class TestConfigsPass(unittest.TestCase):
    def test_load_cfg_success(self) -> None:
        # simply load from the config and this should pass
        self.assertTrue(
            SSLHydraConfig.from_configs(["config=test/integration_test/quick_simclr"]),
            "config must be loaded successfully",
        )


class TestConfigsComposition(unittest.TestCase):
    def test_cfg_composition(self) -> None:
        # compose the configs and check that the model is changed
        cfg = SSLHydraConfig.from_configs(
            [
                "config=test/integration_test/quick_simclr",
                "+config/pretrain/simclr/models=resnext101",
            ]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        self.assertEqual(
            config.MODEL.TRUNK.RESNETS.DEPTH, 101, "config composition failed"
        )


class TestConfigsFailComposition(unittest.TestCase):
    def test_cfg_fail_composition(self) -> None:
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
    def test_cfg_cli_composition(self) -> None:
        # compose the configs and check that the model is changed
        cfg = SSLHydraConfig.from_configs(
            [
                "config=test/integration_test/quick_simclr",
                "+config/pretrain/simclr/models=resnext101",
                "config.MODEL.TRUNK.RESNETS.GROUPS=32",
                "config.MODEL.TRUNK.RESNETS.WIDTH_PER_GROUP=16",
            ]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        self.assertEqual(
            config.MODEL.TRUNK.RESNETS.GROUPS, 32, "config composition failed"
        )
        self.assertEqual(
            config.MODEL.TRUNK.RESNETS.WIDTH_PER_GROUP, 16, "config composition failed"
        )


class TestScalingTypeConfig(unittest.TestCase):
    def test_sqrt_lr_scaling(self) -> None:
        # compose the configs and check that the LR is changed
        cfg = SSLHydraConfig.from_configs(
            [
                "config=test/integration_test/quick_simclr",
                "+config/pretrain/simclr/models=resnext101",
                "config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.auto_scale=True",
                'config.OPTIMIZER.param_schedulers.lr.name="linear"',
                'config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.scaling_type="sqrt"',
            ]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        param_schedulers = config.OPTIMIZER.param_schedulers.lr
        self.assertEqual(0.3 * (0.125**0.5), param_schedulers.end_value)

    def test_linear_lr_scaling(self) -> None:
        # compose the configs and check that the LR is changed
        cfg = SSLHydraConfig.from_configs(
            [
                "config=test/integration_test/quick_simclr",
                "+config/pretrain/simclr/models=resnext101",
                "config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.auto_scale=True",
                'config.OPTIMIZER.param_schedulers.lr.name="linear"',
                'config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.scaling_type="linear"',
            ]
        )
        _, config = convert_to_attrdict(cfg.default_cfg)
        param_schedulers = config.OPTIMIZER.param_schedulers.lr
        self.assertEqual(0.3 * 0.125, param_schedulers.end_value)


class TestConfigsKeyAddition(unittest.TestCase):
    def test_cfg_key_addition(self) -> None:
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
    @parameterized_parallel(BENCHMARK_CONFIGS)
    def test_benchmark_config(self, override):
        logger.warning(f"Loading {override}")
        return hydra_compose(overrides=[override])


class TestPretrainConfigs(unittest.TestCase):
    @parameterized_parallel(PRETRAIN_CONFIGS)
    def test_pretrain_config(self, override):
        logger.warning(f"Loading {override}")
        return hydra_compose(overrides=[override])


class TestIntegrationTestConfigs(unittest.TestCase):
    @parameterized_parallel(INTEGRATION_TEST_CONFIGS)
    def test_integration_test_config(self, override):
        logger.warning(f"Loading {override}")
        return hydra_compose(overrides=[override])
