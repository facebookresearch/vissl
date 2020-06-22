#!/usr/bin/env python3

import unittest

import pkg_resources
from distributed_train import launch_distributed
from parameterized import parameterized
from utils import UNIT_TEST_CONFIGS, SSLHydraConfig
from vissl.ssl_hooks import default_hook_generator
from vissl.utils.hydra_config import convert_to_attrdict


class TaskTest(unittest.TestCase):
    @parameterized.expand(UNIT_TEST_CONFIGS)
    def test_run(self, config_file_path: str):
        """
        Instantiate and run all the test tasks

        Arguments:
            config_file_path {str} -- path to the config for the task to be run
        """

        cfg = SSLHydraConfig.from_configs([config_file_path])
        args, config = convert_to_attrdict(cfg.default_cfg)

        # Complete the data localization at runtime
        config.DATA.TRAIN.DATA_PATHS = [
            pkg_resources.resource_filename(__name__, "test_data")
        ]

        try:
            launch_distributed(config, args, hook_generator=default_hook_generator)
        except Exception as e:
            self.fail(e)
