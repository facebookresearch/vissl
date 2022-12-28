# Portions Copyright (c) Facebook, Inc. and its affiliates.

# Code from: https://github.com/Separius/SimCLRv2-Pytorch/blob/main/resnet.py
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from vissl.config import AttrDict
from vissl.models.trunks import register_model_trunk


BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.9  # == pytorch's default value as well


BLOCK_CONFIG = {
    152: (3, 8, 36, 3),
}


class SUPPORTED_DEPTHS(int, Enum):
    RN152 = 152


class BatchNormRelu(nn.Sequential):
    def __init__(self, num_channels, relu=True):
        super().__init__(
            nn.BatchNorm2d(num_channels, eps=BATCH_NORM_EPSILON),
            nn.ReLU() if relu else nn.Identity(),
        )


def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=bias,
    )


class SelectiveKernel(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio, min_dim=32):
        super().__init__()
        assert sk_ratio > 0.0
        self.main_conv = nn.Sequential(
            conv(in_channels, 2 * out_channels, stride=stride),
            BatchNormRelu(2 * out_channels),
        )
        mid_dim = max(int(out_channels * sk_ratio), min_dim)
        self.mixing_conv = nn.Sequential(
            conv(out_channels, mid_dim, kernel_size=1),
            BatchNormRelu(mid_dim),
            conv(mid_dim, 2 * out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.main_conv(x)
        x = torch.stack(torch.chunk(x, 2, dim=1), dim=0)  # 2, B, C, H, W
        g = x.sum(dim=0).mean(dim=[2, 3], keepdim=True)
        m = self.mixing_conv(g)
        m = torch.stack(torch.chunk(m, 2, dim=1), dim=0)  # 2, B, C, 1, 1
        return (x * F.softmax(m, dim=0)).sum(dim=0)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        if sk_ratio > 0:
            self.shortcut = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                # kernel_size = 2 => padding = 1
                nn.AvgPool2d(kernel_size=2, stride=stride, padding=0),
                conv(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = conv(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        self.bn = BatchNormRelu(out_channels, relu=False)

    def forward(self, x):
        return self.bn(self.shortcut(x))


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(
        self, in_channels, out_channels, stride, sk_ratio=0, use_projection=False
    ):
        super().__init__()
        if use_projection:
            self.projection = Projection(
                in_channels, out_channels * 4, stride, sk_ratio
            )
        else:
            self.projection = nn.Identity()
        ops = [
            conv(in_channels, out_channels, kernel_size=1),
            BatchNormRelu(out_channels),
        ]
        if sk_ratio > 0:
            ops.append(SelectiveKernel(out_channels, out_channels, stride, sk_ratio))
        else:
            ops.append(conv(out_channels, out_channels, stride=stride))
            ops.append(BatchNormRelu(out_channels))
        ops.append(conv(out_channels, out_channels * 4, kernel_size=1))
        ops.append(BatchNormRelu(out_channels * 4, relu=False))
        self.net = nn.Sequential(*ops)

    def forward(self, x):
        shortcut = self.projection(x)
        return F.relu(shortcut + self.net(x))


class Blocks(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, stride, sk_ratio=0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [BottleneckBlock(in_channels, out_channels, stride, sk_ratio, True)]
        )
        self.channels_out = out_channels * BottleneckBlock.expansion
        for _ in range(num_blocks - 1):
            self.blocks.append(
                BottleneckBlock(self.channels_out, out_channels, 1, sk_ratio)
            )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class Stem(nn.Sequential):
    def __init__(self, sk_ratio, width_multiplier):
        ops = []
        channels = 64 * width_multiplier // 2
        if sk_ratio > 0:
            ops.append(conv(3, channels, stride=2))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels))
            ops.append(BatchNormRelu(channels))
            ops.append(conv(channels, channels * 2))
        else:
            ops.append(conv(3, channels * 2, kernel_size=7, stride=2))
        ops.append(BatchNormRelu(channels * 2))
        ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        super().__init__(*ops)


@register_model_trunk("resnet_sk")
class ResNetSelectiveKernel(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super(ResNetSelectiveKernel, self).__init__()
        self.model_config = model_config

        self.trunk_config = self.model_config.TRUNK.RESNETS_SK
        self.depth = SUPPORTED_DEPTHS(self.trunk_config.DEPTH)
        self.width_multiplier = self.trunk_config.WIDTH_MULTIPLIER
        self.sk_ratio = self.trunk_config.SK_RATIO

        layers = BLOCK_CONFIG[self.depth]
        width_multiplier = self.width_multiplier
        sk_ratio = self.sk_ratio

        logging.info(
            f"Building model: ResNet-SK"
            f"-d{self.depth}-{self.width_multiplier}x"
            f"-sk{self.sk_ratio}"
        )

        ops = [Stem(sk_ratio, width_multiplier)]
        channels_in = 64 * width_multiplier
        ops.append(Blocks(layers[0], channels_in, 64 * width_multiplier, 1, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[1], channels_in, 128 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[2], channels_in, 256 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        ops.append(Blocks(layers[3], channels_in, 512 * width_multiplier, 2, sk_ratio))
        channels_in = ops[-1].channels_out
        self.channels_out = channels_in
        self.net = nn.Sequential(*ops)

    def forward(self, x, apply_fc=False):
        h = self.net(x).mean(dim=[2, 3])
        return [h]
