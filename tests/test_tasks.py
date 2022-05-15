# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest

import pkg_resources
import torch
from parameterized import parameterized
from utils import SSLHydraConfig, UNIT_TEST_CONFIGS
from vissl.engines.train import train_main
from vissl.hooks import default_hook_generator
from vissl.utils.checkpoint import get_checkpoint_folder
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
        checkpoint_folder = get_checkpoint_folder(config)

        # Complete the data localization at runtime
        config.DATA.TRAIN.DATA_PATHS = [
            pkg_resources.resource_filename(__name__, "test_data")
        ]

        if torch.distributed.is_initialized():
            # Destroy process groups as torch may be initialized with NCCL, which
            # is incompatible with test_cpu_regnet_moco.yaml
            torch.distributed.destroy_process_group()

        # run training and make sure no exception is raised
        dist_run_id = get_dist_run_id(config, config.DISTRIBUTED.NUM_NODES)
        train_main(
            config,
            dist_run_id=dist_run_id,
            checkpoint_path=None,
            checkpoint_folder=checkpoint_folder,
            local_rank=0,
            node_id=0,
            hook_generator=default_hook_generator,
        )
