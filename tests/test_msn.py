# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from vissl.models import build_model
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import (
    gpu_test,
    in_temporary_directory,
    run_integration_test,
)


class TestMSN(unittest.TestCase):
    """
    Positional dropping used in MSN (https://arxiv.org/pdf/2204.07141.pdf)
    """

    @staticmethod
    def pretraining_config(num_epochs: int = 2, gpu_count: int = 2):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_msn",
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

    @gpu_test(gpu_count=1)
    def test_msn_pos_drop(self) -> None:
        config = self.pretraining_config()
        model = build_model(config["MODEL"], config["OPTIMIZER"])
        x = torch.randn(size=(2, 3, 224, 224))
        y = model(x)[0]
        self.assertEqual(x.shape[0], y.shape[0], "Same batch size")
        self.assertEqual(y.shape[1], 1024, "Number of prototypes")

    @gpu_test(gpu_count=2)
    def test_pretraining(self) -> None:
        config = self.pretraining_config()
        with in_temporary_directory():
            result = run_integration_test(config)
            print(result.get_losses())
