# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head


@register_model_head("mobilenet_v3_head")
class MobileNetV3Head(nn.Module):
    """
    MobileNet-V3 head as done in Torchvision implementation
    """

    def __init__(
        self,
        model_config: AttrDict,
        in_channels: int = 960,
        mid_channels: int = 1280,
        num_classes: int = 1000,
        drop_out: float = 0.2,
        with_bn: bool = False,
    ):
        super().__init__()
        self.with_bn = with_bn
        if self.with_bn:
            self.channel_bn = nn.BatchNorm2d(
                in_channels,
                eps=model_config.HEAD.BATCHNORM_EPS,
                momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
            )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=drop_out, inplace=True),
            nn.Linear(mid_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 2D torch tensor or 4D tensor of shape `N x C x 1 x 1`
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "MLP input should be either a tensor or list containing 1 tensor."
            batch = batch[0]

        if self.with_bn:
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(2).unsqueeze(3)
            assert len(batch.shape) == 4
            out = self.channel_bn(batch)
            out = torch.flatten(out, start_dim=1)
        else:
            out = batch

        out = self.classifier(out)
        return out


@register_model_head("mobilenet_v3_head_timm")
class MobileNetV3HeadTIMM(nn.Module):
    """
    MobileNet-V3 head as done in Torchvision implementation
    """

    def __init__(
        self,
        model_config: AttrDict,
        in_channels: int = 960,
        mid_channels: int = 1280,
        num_classes: int = 1000,
        drop_out: float = 0.2,
        with_bn: bool = False,
    ):
        super().__init__()
        self.with_bn = with_bn
        if self.with_bn:
            self.channel_bn = nn.BatchNorm2d(
                in_channels,
                eps=model_config.HEAD.BATCHNORM_EPS,
                momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
            )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1)),
            nn.Hardswish(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=drop_out, inplace=True),
            nn.Linear(mid_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 2D torch tensor or 4D tensor of shape `N x C x 1 x 1`
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        if isinstance(batch, list):
            assert (
                len(batch) == 1
            ), "MLP input should be either a tensor or list containing 1 tensor."
            batch = batch[0]

        if self.with_bn:
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(2).unsqueeze(3)
            assert len(batch.shape) == 4
            out = self.channel_bn(batch)
        else:
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(2).unsqueeze(3)
            out = batch

        out = self.classifier(out)
        return out
