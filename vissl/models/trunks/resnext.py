#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from enum import Enum

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
from vissl.models.model_helpers import Flatten, _get_norm, parse_out_keys_arg


# For more depths, add the block config here
BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 24, 36, 3),
}


class RESNET_DEPTHS(Enum):
    DEPTH_50 = 50
    DEPTH_101 = 101
    DEPTH_152 = 152
    DEPTH_200 = 200


class INPUT_CHANNEL(Enum):
    lab = 1
    bgr = 3
    rgb = 3


class ResNeXt(nn.Module):
    """
    Wrapper for TorchVison ResNet Model to support different depth and
    width_multiplier. We provide flexibility with LAB input, stride in last
    ResNet block and type of norm (BatchNorm, LayerNorm)
    """

    def __init__(self, model_config, model_name):
        super(ResNeXt, self).__init__()
        self.model_config = model_config

        self.trunk_config = self.model_config.TRUNK.TRUNK_PARAMS.RESNETS
        self.depth = self.trunk_config.DEPTH
        self.width_multiplier = self.trunk_config.WIDTH_MULTIPLIER
        self._norm_layer = _get_norm(self.trunk_config.NORM)
        self.groups = self.trunk_config.GROUPS
        self.zero_init_residual = self.trunk_config.ZERO_INIT_RESIDUAL
        self.width_per_group = self.trunk_config.WIDTH_PER_GROUP

        (n1, n2, n3, n4) = BLOCK_CONFIG[RESNET_DEPTHS(self.depth).value]
        logging.info(
            f"Building model: ResNeXt"
            f"{self.depth}-{self.groups}x{self.width_per_group}d-w{self.width_multiplier}-{self._norm_layer.__name__}"
        )

        model = models.resnet.ResNet(
            block=Bottleneck,
            layers=(n1, n2, n3, n4),
            zero_init_residual=self.zero_init_residual,
            groups=self.groups,
            width_per_group=self.width_per_group,
            norm_layer=self._norm_layer,
        )

        model.inplanes = 64 * self.width_multiplier
        dim_inner = 64 * self.width_multiplier
        # some tasks like Colorization https://arxiv.org/abs/1603.08511 take input
        # as L channel of an LAB image. In that case we change the input channel
        # and re-construct the conv1
        self.input_channels = INPUT_CHANNEL[self.model_config.INPUT_TYPE].value

        model_conv1 = nn.Conv2d(
            self.input_channels,
            model.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model_bn1 = self._norm_layer(model.inplanes)
        model_relu1 = model.relu
        model_maxpool = model.maxpool
        model_layer1 = model._make_layer(Bottleneck, dim_inner, n1)
        model_layer2 = model._make_layer(Bottleneck, dim_inner * 2, n2, stride=2)
        model_layer3 = model._make_layer(Bottleneck, dim_inner * 4, n3, stride=2)
        # For some models like Colorization https://arxiv.org/abs/1603.08511,
        # due to the higher spatial resolution desired for pixel wise task, we
        # support using a different stride. Currently, we know stride=1 and stride=2
        # behavior so support only those.
        if self.trunk_config.LAYER4_STRIDE == 1:
            model_layer4 = model._make_layer(Bottleneck, dim_inner * 8, n4, stride=1)
        else:
            assert self.trunk_config.LAYER4_STRIDE == 2, "Layer4 stride must be 2"
            model_layer4 = model._make_layer(Bottleneck, dim_inner * 8, n4, stride=2)

        model_avgpool = model.avgpool

        # we mapped the layers of resnet model into feature blocks to facilitate
        # feature extraction at various layers of the model. The layers for which
        # to extract features is controlled by out_feat_keys argument in the
        # forward() call.
        self._feature_blocks = nn.ModuleDict(
            [
                ("conv1", model_conv1),
                ("bn1", model_bn1),
                ("conv1_relu", model_relu1),
                ("maxpool", model_maxpool),
                ("layer1", model_layer1),
                ("layer2", model_layer2),
                ("layer3", model_layer3),
                ("layer4", model_layer4),
                ("avgpool", model_avgpool),
                ("flatten", Flatten(1)),
            ]
        )
        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
            "conv1": "conv1_relu",
            "res1": "maxpool",
            "res2": "layer1",
            "res3": "layer2",
            "res4": "layer3",
            "res5": "layer4",
            "res5avg": "avgpool",
            "flatten": "flatten",
        }

    def forward(self, x, out_feat_keys=None):
        mapped_out_feat_keys = []
        if out_feat_keys is not None and len(out_feat_keys) > 0:
            for feat_key in out_feat_keys:
                mapped_out_feat_keys.append(self.feat_eval_mapping[feat_key])

        mapped_out_feat_keys, max_out_feat = parse_out_keys_arg(
            mapped_out_feat_keys, list(self._feature_blocks.keys())
        )
        out_feats = [None] * len(mapped_out_feat_keys)

        feat = x
        # In case of LAB image, we take only "L" channel as input. Split the data
        # along the channel dimension into [L, AB] and keep only L channel.
        if self.model_config.INPUT_TYPE == "lab":
            feat = torch.split(feat, [1, 2], dim=1)[0]

        # Go through the blocks, and save the features as we go
        for i, (feature_name, feature_block) in enumerate(self._feature_blocks.items()):
            feat = feature_block(feat)

            # This feature is requested, store
            if feature_name in mapped_out_feat_keys:
                out_feats[mapped_out_feat_keys.index(feature_name)] = feat

            # Early exit if all the features have been collected
            if i == max_out_feat:
                break
        return out_feats
