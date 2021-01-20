# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
from vissl.models.model_helpers import Flatten, get_trunk_forward_outputs_module_list
from vissl.models.trunks import register_model_trunk
from vissl.utils.hydra_config import AttrDict


@register_model_trunk("alexnet_deepcluster")
class AlexNetDeepCluster(nn.Module):

    # use sobel filter, BN, dim=2
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()

        # first setup the sobel filter
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()

        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        )
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        )
        sobel_filter.bias.data.zero_()
        self.sobel = nn.Sequential(grayscale, sobel_filter)
        for p in self.sobel.parameters():
            p.requires_grad = False

        # Setup the features
        conv1_bn_relu = nn.Sequential(
            nn.Conv2d(2, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        conv2_bn_relu = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
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
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        conv5_bn_relu = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x, out_feat_keys=None):
        feat = x
        # we first apply sobel filter
        feat = self.sobel(feat)
        out_feats = get_trunk_forward_outputs_module_list(
            feat,
            out_feat_keys,
            self._feature_blocks,
            self.all_feat_names,
        )
        return out_feats
