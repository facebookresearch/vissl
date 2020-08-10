# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import unittest

import pkg_resources
from parameterized import parameterized
from utils import UNIT_TEST_CONFIGS, SSLHydraConfig
from vissl.engines.train import train_main
from vissl.ssl_hooks import default_hook_generator
from vissl.utils.hydra_config import convert_to_attrdict
from vissl.utils.misc import get_dist_run_id


logger = logging.getLogger("__name__")


class TaskTest(unittest.TestCase):
    @parameterized.expand(UNIT_TEST_CONFIGS)
    def test_run(self, config_file_path: str):
        """
        Instantiate and run all the test tasks

        Arguments:
            config_file_path {str} -- path to the config for the task to be run
        """
        logger.info(f"Loading {config_file_path}")
        cfg = SSLHydraConfig.from_configs([config_file_path])
        args, config = convert_to_attrdict(cfg.default_cfg)

        # Complete the data localization at runtime
        config.DATA.TRAIN.DATA_PATHS = [
            pkg_resources.resource_filename(__name__, "test_data")
        ]

        try:
            dist_run_id = get_dist_run_id(config, config.DISTRIBUTED.NUM_NODES)
            train_main(
                args,
                config,
                dist_run_id=dist_run_id,
                local_rank=0,
                node_id=0,
                hook_generator=default_hook_generator,
            )
        except Exception as e:
            self.fail(e)
