# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.models.model_helpers import trunc_normal_


@register_model_head("dino_head")
class DINOHead(nn.Module):
    """
    Specific head for training DINO (https://arxiv.org/abs/2104.14294).

    Adapted from the official DINO code base to fit in VISSL:
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    """

    def __init__(
        self,
        model_config: AttrDict,
        in_dim: int,
        num_clusters: List[int],
        use_bn: bool = False,
        normalize_last_layer: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ):
        super().__init__()
        assert num_layers >= 1, "DINO head number of layers should be at least 1"
        assert (
            len(num_clusters) == 1
        ), "DINO head only support one set of clusters for now"

        # Build the MLP
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        # Build the prototypes
        self.prototypes0 = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, num_clusters[0], bias=False)
        )
        self.prototypes0.weight_g.data.fill_(1)
        if normalize_last_layer:
            self.prototypes0.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.prototypes0(x)
        return [x]
