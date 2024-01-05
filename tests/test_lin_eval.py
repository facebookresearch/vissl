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


class TestLinEval(unittest.TestCase):
    @staticmethod
    def _create_config(model_name: str):
        cfg = compose_hydra_configuration(
            [
                "config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear",
                f"+config/test/integration_test/models={model_name}",
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
                "config.LOG_FREQUENCY=1",
                "config.REPRODUCIBILITY.CUDDN_DETERMINISTIC=True",
                "config.OPTIMIZER.num_epochs=10",
                "config.DISTRIBUTED.NUM_PROC_PER_NODE=2",
                "config.MODEL.WEIGHTS_INIT.PARAMS_FILE=''",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    def run_config(self, config, with_memory: bool = False):
        with in_temporary_directory():
            result = run_integration_test(config)
            losses = result.get_losses()
            if with_memory:
                return losses, result.get_peak_memory()
            return losses

    @gpu_test(gpu_count=2)
    def test_linear_eval(self) -> None:
        with in_temporary_directory():
            config = self._create_config("deit_tiny_blocks")
            result = run_integration_test(config)
            losses = result.get_losses()
            print(losses)
