# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestDistillationDINO(unittest.TestCase):
    @staticmethod
    def _create_dino_pretraining_config(num_gpu: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_dino",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.SEED_VALUE=0",
                "config.DISTRIBUTED.NUM_NODES=1",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                "config.LOG_FREQUENCY=1",
                "config.OPTIMIZER.num_epochs=1",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def _create_soft_dino_distillation_config(checkpoint_path: str, num_gpu: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_distillation_dino_2gpus",
                "+config/debugging/pretrain/swav_distillation/optimizer=lars_0_002",
                f"config.DISTILLATION.TEACHER_MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.DATA_LIMIT=40",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.SEED_VALUE=0",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                "config.LOG_FREQUENCY=1",
                "config.OPTIMIZER.num_epochs=2",
                "config.MODEL.SYNC_BN_CONFIG.CONVERT_BN_TO_SYNC_BN=False",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @gpu_test(gpu_count=2)
    def test_dino_soft_distillation(self) -> None:
        with in_temporary_directory() as pretrain_dir:

            # Run a pre-training to have some weights to being with
            pretrain_config = self._create_dino_pretraining_config(num_gpu=2)
            run_integration_test(pretrain_config)
            checkpoint_path = os.path.join(pretrain_dir, "checkpoint.torch")

            # Distillation to a DEIT
            with in_temporary_directory():
                distill_config = self._create_soft_dino_distillation_config(
                    checkpoint_path=checkpoint_path,
                    num_gpu=2,
                )

                # Run distillation
                result = run_integration_test(distill_config)
                losses = result.get_losses()
                print(losses)
                self.assertTrue(10, len(losses))
                self.assertGreater(losses[0], losses[-1])
