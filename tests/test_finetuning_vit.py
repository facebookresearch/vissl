# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest

from classy_vision.optim import build_optimizer_schedulers
from vissl.models import build_model
from vissl.optimizers import get_optimizer_param_groups
from vissl.optimizers.param_groups.layer_decay import (
    get_resnet_param_depth,
    get_vit_param_depth,
)
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.test_utils import gpu_test


class TestFineTuningVit(unittest.TestCase):
    @staticmethod
    def _create_pretraining_config(name: str):
        cfg = compose_hydra_configuration(
            [
                f"config=test/integration_test/{name}",
                "config.OPTIMIZER.param_group_constructor=lr_decay",
            ]
        )
        args, config = convert_to_attrdict(cfg)
        return config

    @gpu_test(gpu_count=1)
    def test_vit_param_groups(self) -> None:
        config = self._create_pretraining_config("quick_dino")
        model = build_model(config["MODEL"], config["OPTIMIZER"])
        for n, _p in model.named_parameters():
            print(n, get_vit_param_depth(n, num_layers=12))

        for layer_name, expected_depth in [
            ("trunk.patch_embed.proj.bias", 0),
            ("trunk.blocks.0.norm1.weight", 1),
            ("trunk.blocks.8.norm1.weight", 9),
            ("trunk.blocks.11.attn.qkv.weight", 12),
        ]:
            depth = get_vit_param_depth(layer_name, num_layers=12)
            self.assertEqual(expected_depth, depth)

        optimizer_schedulers = build_optimizer_schedulers(config["OPTIMIZER"])
        param_groups = get_optimizer_param_groups(
            model=model,
            model_config=config["MODEL"],
            optimizer_config=config["OPTIMIZER"],
            optimizer_schedulers=optimizer_schedulers,
        )
        for g in param_groups:
            for k in ["lr", "weight_decay"]:
                print(k, g[k])

    @gpu_test(gpu_count=1)
    def test_resnet_param_groups(self) -> None:
        config = self._create_pretraining_config("quick_swav")
        config.OPTIMIZER.regularize_bias = False
        model = build_model(config["MODEL"], config["OPTIMIZER"])
        for n, _p in model.named_parameters():
            print(n, get_resnet_param_depth(n, num_layers=4))

        for layer_name, expected_depth in [
            ("trunk._feature_blocks.conv1.weight", 0),
            ("trunk._feature_blocks.bn1.weight", 0),
            ("trunk._feature_blocks.bn1.bias", 0),
            ("trunk._feature_blocks.layer2.0.downsample.1.bias", 2),
            ("trunk._feature_blocks.layer4.1.conv3.weight", 4),
        ]:
            depth = get_resnet_param_depth(layer_name, num_layers=4)
            self.assertEqual(expected_depth, depth)

        optimizer_schedulers = build_optimizer_schedulers(config["OPTIMIZER"])
        param_groups = get_optimizer_param_groups(
            model=model,
            model_config=config["MODEL"],
            optimizer_config=config["OPTIMIZER"],
            optimizer_schedulers=optimizer_schedulers,
        )
        for g in param_groups:
            for k in ["lr", "weight_decay"]:
                print(k, g[k])
