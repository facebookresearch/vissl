# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/gidariss/FeatureLearningRotNet

import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.model_helpers import Flatten, get_trunk_forward_outputs_module_list
from vissl.models.trunks import register_model_trunk


@register_model_trunk("alexnet_rotnet")
class AlexNetRotNet(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()

        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        flatten = Flatten()

        self._feature_blocks = nn.ModuleList(
            [conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5, flatten]
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

    def forward(self, x, out_feat_keys=None):
        feat = x
        out_feats = get_trunk_forward_outputs_module_list(
            feat, out_feat_keys, self._feature_blocks, self.all_feat_names
        )
        return out_feats
