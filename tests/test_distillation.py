# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from hydra.experimental import compose, initialize_config_module
from vissl.utils.hydra_config import convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestDistillation(unittest.TestCase):
    @staticmethod
    def _create_pretraining_config(num_gpu: int = 2):
        with initialize_config_module(config_module="vissl.config"):
            cfg = compose(
                "defaults",
                overrides=[
                    "config=test/integration_test/quick_swav",
                    "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                    "config.DATA.TRAIN.DATA_LIMIT=40",
                    "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                    "config.SEED_VALUE=0",
                    "config.LOSS.swav_loss.epsilon=0.03",
                    f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                    "config.LOG_FREQUENCY=1",
                ],
            )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def _create_distillation_config(checkpoint_path: str, num_gpu: int = 2):
        with initialize_config_module(config_module="vissl.config"):
            cfg = compose(
                "defaults",
                overrides=[
                    "config=debugging/distillation/distill_rn50_to_rn50_mse_trunk",
                    f"config.DISTILLATION.TEACHER_MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                    "config.TEST_MODEL=False",
                    "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                    "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                    "config.DATA.TRAIN.LABEL_TYPE=sample_index",
                    "config.DATA.TRAIN.DATA_LIMIT=40",
                    "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                    "config.SEED_VALUE=0",
                    "config.DISTRIBUTED.NUM_NODES=1",
                    f"config.DISTRIBUTED.NUM_PROC_PER_NODE={num_gpu}",
                    "config.LOG_FREQUENCY=1",
                    "config.OPTIMIZER.num_epochs=2",
                ],
            )
        args, config = convert_to_attrdict(cfg)
        return config

    @gpu_test(gpu_count=2)
    def test_soft_distillation(self) -> None:
        with in_temporary_directory() as pretrain_dir:

            # Run a pre-training to have some weights to being with
            pretrain_config = self._create_pretraining_config(num_gpu=2)
            run_integration_test(pretrain_config)
            checkpoint_path = os.path.join(pretrain_dir, "checkpoint.torch")

            # Create a separate directly in which to run the soft distillation
            with in_temporary_directory():
                finetune_config = self._create_distillation_config(
                    checkpoint_path=checkpoint_path, num_gpu=2
                )
                result = run_integration_test(finetune_config)
                losses = result.get_losses()
                print(losses)
