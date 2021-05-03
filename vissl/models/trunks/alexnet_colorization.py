# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.model_helpers import Flatten, get_trunk_forward_outputs_module_list
from vissl.models.trunks import register_model_trunk


@register_model_trunk("alexnet_colorization")
class AlexNetColorization(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        conv1_bn_relu = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        conv2_bn_relu = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        conv3_bn_relu = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        conv4_bn_relu = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        conv5_bn_relu = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        flatten = Flatten()

        self._feature_blocks = nn.ModuleList(
            [
                conv1_bn_relu,
                pool1,
                conv2_bn_relu,
                pool2,
                conv3_bn_relu,
                conv4_bn_relu,
                conv5_bn_relu,
                pool3,
                flatten,
            ]
        )
        self.all_feat_names = [
            "conv1",
            "pool1",
            "conv2",
            "pool2",
            "conv3",
            "conv4",
            "conv5",
            "pool5",
            "flatten",
        ]
        assert len(self.all_feat_names) == len(self._feature_blocks)
        assert (
            model_config.INPUT_TYPE == "lab"
        ), "AlexNet Colorization model takes LAB image only"

    def forward(self, x, out_feat_keys=None):
        feat = x
        # In case of LAB image, we take only "L" channel as input. Split the data
        # along the channel dimension into [L, AB] and keep only L channel.
        feat = torch.split(feat, [1, 2], dim=1)[0]
        out_feats = get_trunk_forward_outputs_module_list(
            feat, out_feat_keys, self._feature_blocks, self.all_feat_names
        )
        return out_feats
