# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import torch
from vissl.models import build_model
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import gpu_test


class TestSwinT(unittest.TestCase):
    """
    Test the VISSL SwinT implementation against the official implementation of:
    https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py

    As of today, April 2022, the VISSL implementation is more flexible and allows
    input multiple resolutions without requiring up-sampling / down-sampling.
    """

    @staticmethod
    def official_swin_transformer_config():
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_dino_swin_t",
                "config.MODEL.TRUNK.NAME=swin_transformer_official",
                "config.MODEL.TRUNK.SWIN_TRANSFORMER.DROP_PATH_RATE=0.0",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def vissl_swin_transformer_config():
        cfg = compose_hydra_configuration(
            [
                "config=test/integration_test/quick_dino_swin_t",
                "config.MODEL.TRUNK.NAME=swin_transformer",
                "config.MODEL.TRUNK.SWIN_TRANSFORMER.DROP_PATH_RATE=0.0",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @staticmethod
    def build_eval_model(config, seed=0):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        model = build_model(config["MODEL"], config["OPTIMIZER"]).cuda()
        model.eval()
        return model

    @gpu_test(gpu_count=1)
    def test_vissl_implementation_against_official(self) -> None:
        with torch.no_grad():
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            x = torch.randn(size=(2, 3, 224, 224)).cuda()

            ref_model = self.build_eval_model(
                config=self.official_swin_transformer_config(), seed=0
            )
            vissl_model = self.build_eval_model(
                config=self.vissl_swin_transformer_config(),
                seed=0,
            )
            ref = ref_model.trunk(x)[0]
            out = vissl_model.trunk(x)[0]
            self.assertEqual(ref.shape, out.shape)
            self.assertTrue(torch.allclose(ref, out, atol=1e-5))

    @gpu_test(gpu_count=1)
    def test_vissl_implementation_support_multiple_resolutions(self) -> None:
        with torch.no_grad():
            config = self.vissl_swin_transformer_config()
            model = build_model(config["MODEL"], config["OPTIMIZER"]).cuda()

            x1 = torch.randn(size=(2, 3, 224, 224)).cuda()
            y1 = model.trunk(x1)[0]
            self.assertEqual(y1.shape, torch.Size([2, 768]))

            x2 = torch.randn(size=(2, 3, 96, 96)).cuda()
            y2 = model.trunk(x2)[0]
            self.assertEqual(y2.shape, torch.Size([2, 768]))

            x3 = torch.randn(size=(2, 3, 95, 95)).cuda()
            y3 = model.trunk(x3)[0]
            self.assertEqual(y3.shape, torch.Size([2, 768]))
