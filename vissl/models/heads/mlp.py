#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        model_config,
        dims,
        use_bn=False,
        use_relu=False,
        use_dropout=False,
        use_bias=True,
    ):
        super().__init__()
        layers = []
        last_dim = dims[0]
        for dim in dims[1:]:
            layers.append(nn.Linear(last_dim, dim, bias=use_bias))
            if use_bn:
                layers.append(
                    nn.BatchNorm1d(
                        dim,
                        eps=model_config.HEAD.BATCHNORM_EPS,
                        momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
                    )
                )
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
                last_dim = dim
            if use_dropout:
                layers.append(nn.Dropout())
        self.clf = nn.Sequential(*layers)

    def forward(self, batch):
        out = self.clf(batch)
        return out
