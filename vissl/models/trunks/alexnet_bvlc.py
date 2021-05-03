# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.model_helpers import Flatten, get_trunk_forward_outputs_module_list
from vissl.models.trunks import register_model_trunk


@register_model_trunk("alexnet_bvlc")
class AlexNetBvlc(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()

        conv1_relu = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), nn.ReLU(inplace=True)
        )

        lrn_pool1 = nn.Sequential(
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        conv2_relu = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
        )

        lrn_pool2 = nn.Sequential(
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        conv3_relu = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        conv4_relu = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
        )

        conv5_relu = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
        )

        maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        flatten = Flatten()

        self._feature_blocks = nn.ModuleList(
            [
                conv1_relu,
                lrn_pool1,
                conv2_relu,
                lrn_pool2,
                conv3_relu,
                conv4_relu,
                conv5_relu,
                maxpool3,
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

    def forward(self, x, out_feat_keys=None):
        feat = x
        out_feats = get_trunk_forward_outputs_module_list(
            feat,
            out_feat_keys,
            self._feature_blocks,
            self.all_feat_names,
            use_checkpointing=False,
        )
        return out_feats
