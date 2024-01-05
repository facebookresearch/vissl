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
    """
    Integrations tests that should be run on 8 GPUs nodes.

    Tests that the RegNet10B trained for SEER still works:
    https://arxiv.org/abs/2202.08360
    """

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
    def test_regnet_10b_swav_pretraining(self) -> None:
        with in_temporary_directory():
            config = self._create_10B_pretrain_config(
                num_gpus=8, num_steps=2, batch_size=4
            )
            results = run_integration_test(config)
            losses = results.get_losses()
            print(losses)
            self.assertEqual(len(losses), 2)

    @staticmethod
    def _create_10B_evaluation_config(
        num_gpus: int, num_steps: int, batch_size: int, path_to_sliced_checkpoint: str
    ):
        data_limit = num_steps * batch_size * num_gpus
        cfg = compose_hydra_configuration(
            [
                "config=benchmark/linear_image_classification/clevr_count/eval_resnet_8gpu_transfer_clevr_count_linear",
                "+config/benchmark/linear_image_classification/clevr_count/models=regnet10B",
                f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={path_to_sliced_checkpoint}",
                "config.MODEL.AMP_PARAMS.USE_AMP=True",
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                "config.OPTIMIZER.num_epochs=1",
                "config.LOG_FREQUENCY=1",
                # Testing on fake images
                "config.DATA.TRAIN.DATA_SOURCES=[synthetic]",
                "config.DATA.TRAIN.LABEL_SOURCES=[synthetic]",
                "config.DATA.TRAIN.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TRAIN.USE_DEBUGGING_SAMPLER=True",
                "config.DATA.TEST.DATA_SOURCES=[synthetic]",
                "config.DATA.TEST.LABEL_SOURCES=[synthetic]",
                "config.DATA.TEST.RANDOM_SYNTHETIC_IMAGES=True",
                "config.DATA.TEST.USE_DEBUGGING_SAMPLER=True",
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
    def test_regnet_10b_evaluation(self) -> None:
        with in_temporary_directory():
            cp_path = "/checkpoint/qduval/vissl/seer/regnet10B_sliced/model_iteration124500_sliced.torch"
            config = self._create_10B_evaluation_config(
                num_gpus=8, num_steps=2, batch_size=4, path_to_sliced_checkpoint=cp_path
            )
            results = run_integration_test(config)
            losses = results.get_losses()
            print(losses)
            self.assertGreater(len(losses), 0)
