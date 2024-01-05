# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from vissl.config import AttrDict
from vissl.models.heads import LinearEvalMLP, MLP


class TestMLP(unittest.TestCase):
    """
    Unit test to verify that correct construction of MLP layers
    and linear evaluation MLP layers
    """

    MODEL_CONFIG = AttrDict(
        {
            "HEAD": {
                "BATCHNORM_EPS": 1e-6,
                "BATCHNORM_MOMENTUM": 0.99,
                "PARAMS_MULTIPLIER": 1.0,
            }
        }
    )

    def test_mlp(self) -> None:
        mlp = MLP(self.MODEL_CONFIG, dims=[2048, 100])

        x = torch.randn(size=(4, 2048))
        out = mlp(x)
        assert out.shape == torch.Size([4, 100])

        x = torch.randn(size=(1, 2048))
        out = mlp(x)
        assert out.shape == torch.Size([1, 100])

    def test_mlp_reshaping(self) -> None:
        mlp = MLP(self.MODEL_CONFIG, dims=[2048, 100])

        x = torch.randn(size=(1, 2048, 1, 1))
        out = mlp(x)
        assert out.shape == torch.Size([1, 100])

    def test_mlp_catch_bad_shapes(self) -> None:
        mlp = MLP(self.MODEL_CONFIG, dims=[2048, 100])

        x = torch.randn(size=(1, 2048, 2, 1))
        with self.assertRaises(AssertionError) as context:
            mlp(x)
        assert context.exception is not None

    def test_eval_mlp_shape(self) -> None:
        eval_mlp = LinearEvalMLP(
            self.MODEL_CONFIG, in_channels=2048, dims=[2048 * 2 * 2, 1000]
        )

        resnet_feature_map = torch.randn(size=(4, 2048, 2, 2))
        out = eval_mlp(resnet_feature_map)
        assert out.shape == torch.Size([4, 1000])

        resnet_feature_map = torch.randn(size=(1, 2048, 2, 2))
        out = eval_mlp(resnet_feature_map)
        assert out.shape == torch.Size([1, 1000])
