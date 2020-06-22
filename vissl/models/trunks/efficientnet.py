#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from collections import OrderedDict

import torch.nn as nn
from classy_vision.models.efficientnet import (
    MODEL_PARAMS,
    EfficientNet as ClassyEfficientNet,
)
from vissl.models.model_helpers import Flatten, parse_out_keys_arg


class EfficientNet(nn.Module):
    """
    Wrapper for ClassyVision EfficientNet model so we can map layers into feature
    blocks to facilitate feature extraction and benchmarking at several layers.
    """

    def __init__(self, model_config, model_name):
        super(EfficientNet, self).__init__()
        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"

        trunk_config = model_config.TRUNK.TRUNK_PARAMS.EFFICIENT_NETS
        assert "model_version" in trunk_config, "Please specify EfficientNet version"
        model_version = trunk_config["model_version"]
        assert (
            model_version in MODEL_PARAMS.keys()
        ), f"EfficientNet {model_version} not found"
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
        self.relu_fn = model.relu_fn
        self.dropout = model.dropout

        # we mapped the layers of model into feature blocks to facilitate
        # feature extraction at various layers of the model. The layers for which
        # to extract features is controlled by out_feat_keys argument in the
        # forward() call.
        conv1 = nn.Sequential(model.conv_stem, model.bn0)
        feature_blocks = [conv1]

        # since the number of blocks scales with different configuration of
        # model, we look at the names of the children and count the blocks
        all_blocks = model.blocks.named_children()
        block_names = [blk[0].strip().split("-")[0] for blk in all_blocks]
        block_counts = OrderedDict()
        for item in block_names:
            if item in block_counts:
                block_counts[item] += 1
            else:
                block_counts[item] = 1
        count = 0
        for _, num in block_counts.items():
            seq_block = model.blocks[count : (count + num)]
            feature_blocks.append(seq_block)
            count += num

        feature_blocks.append(nn.Sequential(model.conv_head, model.bn1))
        feature_blocks.append(model.avg_pooling)
        feature_blocks.append(Flatten(1))
        if model.dropout:
            feature_blocks.append(model.dropout)

        self._feature_blocks = nn.ModuleList(feature_blocks)
        self.all_feat_names = [
            "conv1",
            "block0",
            "block1",
            "block2",
            "block3",
            "block4",
            "block5",
            "block6",
            "conv_final",
            "avgpool",
            "flatten",
        ]
        if model.dropout:
            self.all_feat_names.append("dropout")

    def forward(self, x, out_feat_keys=None):
        out_feat_keys, max_out_feat = parse_out_keys_arg(
            out_feat_keys, self.all_feat_names
        )
        out_feats = [None] * len(out_feat_keys)

        feat = x
        block_num = 0
        for f in range(max_out_feat + 1):
            key = self.all_feat_names[f]
            if "block" in key:
                for idx in range(len(self._feature_blocks[f])):
                    drop_connect_rate = self.drop_connect_rate
                    if self.drop_connect_rate:
                        drop_connect_rate *= float(block_num) / self.num_blocks
                    feat = self._feature_blocks[f][idx](
                        feat, drop_connect_rate=drop_connect_rate
                    )
                    block_num += 1
            elif "conv" in key:
                feat = self.relu_fn(self._feature_blocks[f](feat))
            else:
                feat = self._feature_blocks[f](feat)
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat
        return out_feats
