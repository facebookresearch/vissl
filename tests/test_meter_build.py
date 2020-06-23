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
from utils import ROOT_CONFIGS, SSLHydraConfig
from vissl.ssl_tasks import build_task
from vissl.utils.hydra_config import convert_to_attrdict


logger = logging.getLogger("__name__")


class TestRootConfigsMeterBuild(unittest.TestCase):
    @parameterized.expand(ROOT_CONFIGS)
    def test_meter_build(self, filepath):
        logger.info(f"Loading {filepath}")
        cfg = SSLHydraConfig.from_configs([filepath])
        _, config = convert_to_attrdict(cfg.default_cfg)
        meters = build_task(config)._build_meters()
        self.assertGreaterEqual(len(meters), 0, "Failed to build meters")
