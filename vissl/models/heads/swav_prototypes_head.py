# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn


class SwAVPrototypesHead(nn.Module):
    def __init__(self, model_config, dims, use_bn, nmb_clusters, use_bias=True):
        super().__init__()
        # build the projection head
        layers = []
        last_dim = dims[0]
        for i, dim in enumerate(dims[1:]):
            layers.append(nn.Linear(last_dim, dim, bias=use_bias))
            if i == len(dims) - 2:
                break
            if use_bn:
                layers.append(
                    nn.BatchNorm1d(
                        dim,
                        eps=model_config.HEAD.BATCHNORM_EPS,
                        momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
                    )
                )
            layers.append(nn.ReLU(inplace=True))
            last_dim = dim
        self.projection_head = nn.Sequential(*layers)

        # prototypes (i.e. centroids) layers
        if len(nmb_clusters) > 0:
            self.nmb_heads = len(nmb_clusters)
            for i, k in enumerate(nmb_clusters):
                self.add_module(
                    "prototypes" + str(i), nn.Linear(dims[-1], k, bias=False)
                )
        else:
            self.nmb_heads = 0

    def forward(self, x):
        x = self.projection_head(x)

        x = nn.functional.normalize(x, dim=1, p=2)

        out = [x]
        if self.nmb_heads > 0:
            for i in range(self.nmb_heads):
                out.append(getattr(self, "prototypes" + str(i))(x))

        return out
