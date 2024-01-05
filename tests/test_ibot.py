# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest

import torch
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestIBOT(unittest.TestCase):
    """
    Test for IBOT (https://arxiv.org/pdf/2111.07832.pdf)
    """

    @staticmethod
    def create_pretraining_config(num_epochs: int = 2, gpu_count: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_ibot",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                f"config.OPTIMIZER.num_epochs={num_epochs}",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={gpu_count}",
            ]
        )
        _args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def create_linear_eval_config(checkpoint_path: str, gpu_count: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_eval_in1k_linear.yaml",
                "+config/benchmark/linear_image_classification/imagenet1k/models=ibot_deit_s16",
                f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={checkpoint_path}",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={gpu_count}",
                # Datasets
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TEST.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TEST.DATA_LIMIT=32",
                "config.DATA.TEST.USE_DEBUGGING_SAMPLER=True",
                # To get the logs reliably
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                "config.OPTIMIZER.num_epochs=2",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @gpu_test(gpu_count=2)
    def test_training_with_preemption_then_evaluation(self) -> None:
        config = self.create_pretraining_config()
        with in_temporary_directory() as pretrain_dir:
            result = run_integration_test(config)
            losses_before = result.get_losses()

            # Check content of pre-training directory after run is done
            dir_content = os.listdir(pretrain_dir)
            self.assertIn("model_final_checkpoint_phase1.torch", dir_content)

            # Check the content of the checkpoint
            checkpoint = torch.load("model_final_checkpoint_phase1.torch")
            self.assertIn("loss", checkpoint)
            self.assertIn("criterion", checkpoint["loss"].keys())
            self.assertIn("teacher", checkpoint["loss"].keys())

            # Simulate a preemption by removing the last checkpoint
            # and then restarting the training
            os.remove("model_final_checkpoint_phase1.torch")
            os.remove("checkpoint.torch")
            os.remove("log.txt")
            result = run_integration_test(config)
            losses_after = result.get_losses()

            # Check that the preemption leads to restart at last checkpoint
            # and check that the losses match despite the preemption
            print("Losses before:", losses_before)
            print("Losses after:", losses_after)
            self.assertEqual(8, len(losses_before))
            self.assertEqual(4, len(losses_after))
            self.assertAlmostEqual(losses_after[-1], losses_before[-1], places=5)

            # Now load the checkpoint for evaluation
            with in_temporary_directory():
                eval_config = self.create_linear_eval_config(
                    gpu_count=2,
                    checkpoint_path=os.path.join(pretrain_dir, "checkpoint.torch"),
                )
                result = run_integration_test(eval_config)
                losses = result.get_losses()
                self.assertGreater(len(losses), 0)
