# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
from vissl.models.heads.mlp import MLP


class LinearEvalMLP(nn.Module):
    def __init__(self, model_config, in_channels, dims):
        super().__init__()
        self.channel_bn = nn.BatchNorm2d(
            in_channels,
            eps=model_config.HEAD.BATCHNORM_EPS,
            momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
        )

        self.clf = MLP(model_config, dims)

    def forward(self, batch):
        out = self.channel_bn(batch)
        out = torch.flatten(out, start_dim=1)
        out = self.clf(out)
        return out
