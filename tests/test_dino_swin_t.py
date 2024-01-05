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


class TestDINO_SwinT(unittest.TestCase):
    @staticmethod
    def _create_dino_pretraining_config(
        with_mixed_precision: bool, gpu_count: int = 2, num_epochs: int = 4
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_dino_swin_t",
                "config.SEED_VALUE=0",
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=4",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.DATA_LIMIT=32",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                f"config.OPTIMIZER.num_epochs={num_epochs}",
                f"config.DISTRIBUTED.NUM_PROC_PER_NODE={gpu_count}",
                # Options to override to get FSDP
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                f"config.MODEL.AMP_PARAMS.USE_AMP={with_mixed_precision}",
                "config.OPTIMIZER.construct_single_param_group_only=True",
                "config.MODEL.FSDP_CONFIG.AUTO_WRAP_THRESHOLD=0",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def _create_dino_linear_eval_config(checkpoint_path: str, gpu_count: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_eval_in1k_linear",
                "+config/benchmark/linear_image_classification/imagenet1k/models=swint_tiny",
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

    def run_config(self, config):
        with in_temporary_directory():
            result = run_integration_test(config)
            return result.get_losses()

    @gpu_test(gpu_count=2)
    def test_pretraining_and_evaluation(self) -> None:
        with in_temporary_directory() as pretrain_dir:
            config = self._create_dino_pretraining_config(
                with_mixed_precision=True, gpu_count=2, num_epochs=1
            )
            result = run_integration_test(config)
            ddp_losses = result.get_losses()
            self.assertGreater(len(ddp_losses), 0)

            eval_config = self._create_dino_linear_eval_config(
                checkpoint_path=os.path.join(pretrain_dir, "checkpoint.torch"),
                gpu_count=2,
            )
            eval_losses = self.run_config(eval_config)
            print(eval_losses)

    @gpu_test(gpu_count=2)
    def test_pretraining_preemption(self) -> None:
        with in_temporary_directory() as temp_dir:
            config = self._create_dino_pretraining_config(
                with_mixed_precision=False, gpu_count=2
            )
            config.MODEL.TRUNK.SWIN_TRANSFORMER.DROP_PATH_RATE = 0.0
            result = run_integration_test(config)
            losses_before = result.get_losses()

            temp_dir_content = os.listdir(temp_dir)
            self.assertIn("model_final_checkpoint_phase3.torch", temp_dir_content)
            os.remove("model_final_checkpoint_phase3.torch")
            os.remove("checkpoint.torch")
            os.remove("log.txt")

            result = run_integration_test(config)
            losses_after = result.get_losses()
            print(losses_before)
            print(losses_after)
            self.assertAlmostEqual(losses_after[-1], losses_before[-1], places=5)
