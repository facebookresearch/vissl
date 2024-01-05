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


class TestSwAV(unittest.TestCase):
    @staticmethod
    def _create_pretraining_config(
        with_mixed_precision: bool,
        gpu_count: int = 2,
        num_epochs: int = 4,
    ):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_swav",
                "+config/pretrain/swav/models=resnet18",
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
                "config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch",
                f"config.MODEL.AMP_PARAMS.USE_AMP={with_mixed_precision}",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @gpu_test(gpu_count=2)
    def test_pretraining_and_evaluation(self) -> None:
        with in_temporary_directory():
            config = self._create_pretraining_config(
                with_mixed_precision=True, gpu_count=2, num_epochs=1
            )
            result = run_integration_test(config)
            ddp_losses = result.get_losses()
            self.assertGreater(len(ddp_losses), 0)
