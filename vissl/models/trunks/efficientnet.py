# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List

import torch
import torch.nn as nn
from classy_vision.models.efficientnet import (
    EfficientNet as ClassyEfficientNet,
    MODEL_PARAMS,
)
from vissl.config import AttrDict
from vissl.models.model_helpers import Flatten, parse_out_keys_arg, Wrap
from vissl.models.trunks import register_model_trunk


@register_model_trunk("efficientnet")
class EfficientNet(nn.Module):
    """
    Wrapper for ClassyVision EfficientNet model so we can map layers into feature
    blocks to facilitate feature extraction and benchmarking at several layers.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(EfficientNet, self).__init__()
        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"

        trunk_config = model_config.TRUNK.EFFICIENT_NETS
        assert "model_version" in trunk_config, "Please specify EfficientNet version"
        model_version = trunk_config["model_version"]
        model_params = MODEL_PARAMS[model_version]

        trunk_config["model_params"] = model_params
        trunk_config.pop("model_version")
        # we don't use the FC constructed with num_classes. This param is required
        # to build the model in Classy Vision hence we pass the default value.
        trunk_config["num_classes"] = 1000
        logging.info(f"Building model: EfficientNet-{model_version}")
        model = ClassyEfficientNet(**trunk_config)

        self.drop_connect_rate = model.drop_connect_rate
        self.num_blocks = len(model.blocks)
        self.dropout = model.dropout
        self.activation = Wrap(model.relu_fn)  # using swish, not relu actually

        # We map the layers of model into feature blocks to facilitate
        # feature extraction at various layers of the model. The layers for which
        # to extract features is controlled by out_feat_keys argument in the
        # forward() call.
        # - Stem
        feature_blocks = [
            ["conv1", nn.Sequential(model.conv_stem, model.bn0, self.activation)]
        ]

        # - Mobile Inverted Residual Bottleneck blocks
        feature_blocks.extend(
            [[f"block{i}", v] for i, v in enumerate(model.blocks.children())]
        )

        # - Conv Head + Pooling
        feature_blocks.extend(
            [
                [
                    "conv_final",
                    nn.Sequential(model.conv_head, model.bn1, self.activation),
                ],
                ["avgpool", model.avg_pooling],
                ["flatten", Flatten(1)],
            ]
        )

        if model.dropout:
            feature_blocks.append(["dropout", model.dropout])

        # Consolidate into one indexable trunk
        self._feature_blocks = nn.ModuleDict(feature_blocks)
        self.all_feat_names = list(self._feature_blocks.keys())

    def forward(self, x: torch.Tensor, out_feat_keys: List[str] = None):
        out_feat_keys, max_out_feat = parse_out_keys_arg(
            out_feat_keys, self.all_feat_names
        )
        out_feats = [None] * len(out_feat_keys)
        feat = x

        # Walk through the EfficientNet, block by block
        blocks = iter(self._feature_blocks.named_children())

        # - First block is always the stem
        stem_name, stem_block = next(blocks)
        feat = stem_block(feat)
        if stem_name in out_feat_keys:
            out_feats[out_feat_keys.index(stem_name)] = feat

        # - Next go through all the MIRB, then the eventual conv and pooling
        for i, (feature_name, feature_block) in enumerate(blocks):
            if "block" in feature_name:
                # -- MIRB blocks (needs ad-hoc drop connect rate)
                drop_connect_rate = self.drop_connect_rate
                if self.drop_connect_rate:
                    drop_connect_rate *= float(i) / self.num_blocks
                feat = feature_block(feat, drop_connect_rate=drop_connect_rate)
            else:
                # -- Conv, Pooling (simple forward)
                feat = feature_block(feat)

            # If requested, store the feature
            if feature_name in out_feat_keys:
                out_feats[out_feat_keys.index(feature_name)] = feat

            # Early exit if all the features have been collected
            if i == max_out_feat:
                break

        return out_feats
