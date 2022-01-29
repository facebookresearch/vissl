# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestRegnet10B(unittest.TestCase):
    @staticmethod
    def _create_10B_pretrain_config(num_gpus: int, num_steps: int, batch_size: int):
        data_limit = num_steps * batch_size * num_gpus
        cfg = compose_hydra_configuration(
            [
                "config=pretrain/swav/swav_8node_resnet",
                "+config/pretrain/seer/models=regnet10B",
                "config.OPTIMIZER.num_epochs=1",
                "config.LOG_FREQUENCY=1",
                # Testing on fake images
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                # Disable overlap communication and computation for test
                "config.MODEL.FSDP_CONFIG.FORCE_SYNC_CUDA=True",
                # Testing on 8 V100 32GB GPU only
                f"config.DATA.TRAIN.BATCHSIZE_PER_REPLICA={batch_size}",
                f"config.DATA.TRAIN.DATA_LIMIT={data_limit}",
                "config.DISTRIBUTED.NUM_NODES=1",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpus}",
                "config.DISTRIBUTED.RUN_ID=auto",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @gpu_test(gpu_count=8)
    def test_regnet_10b_swav_pretraining(self):
        with in_temporary_directory():
            config = self._create_10B_pretrain_config(
                num_gpus=8, num_steps=2, batch_size=4
            )
            results = run_integration_test(config)
            losses = results.get_losses()
            print(losses)
            self.assertEqual(len(losses), 2)
