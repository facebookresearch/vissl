# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from vissl.models import build_model
from vissl.optimizers import *  # noqa
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict


class TestResNeXt(unittest.TestCase):
    """
    Test the Regnet FSDP model in comparison with the DDP Regnet
    to verify that both converge to the same losses
    """

    @staticmethod
    def _create_model_config(depth: int):
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_knn",
                f"config.MODEL.TRUNK.RESNETS.DEPTH={depth}",
                "config.MODEL.HEAD.PARAMS=[['identity', {}]]",
            ],
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def _get_number_parameters(model):
        return sum(p.numel() for p in model.parameters())

    def test_valid_depths(self) -> None:
        config = self._create_model_config(depth=18)
        model = build_model(config["MODEL"], config["OPTIMIZER"])
        print(self._get_number_parameters(model))

        config = self._create_model_config(depth=34)
        model = build_model(config["MODEL"], config["OPTIMIZER"])
        print(self._get_number_parameters(model))

        config = self._create_model_config(depth=50)
        model = build_model(config["MODEL"], config["OPTIMIZER"])
        print(self._get_number_parameters(model))
