# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

from parameterized import parameterized
from utils import ROOT_LOSS_CONFIGS, SSLHydraConfig
from vissl.trainer.train_task import SelfSupervisionTask
from vissl.utils.hydra_config import convert_to_attrdict


logger = logging.getLogger("__name__")


class TestRootConfigsMeterBuild(unittest.TestCase):
    @parameterized.expand(ROOT_LOSS_CONFIGS)
    def test_meter_build(self, filepath):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs([filepath])
        _, config = convert_to_attrdict(cfg.default_cfg)
        meters = SelfSupervisionTask.from_config(config)._build_meters()
        self.assertGreaterEqual(len(meters), 0, "Failed to build meters")
