# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.models.heads.ibot_head import CustomSequential
from vissl.models.model_helpers import trunc_normal_


@register_model_head("ibot_distillation_head")
class IBOTDistillationHead(nn.Module):
    """
    Specific head for distilling iBOT (https://arxiv.org/pdf/2111.07832.pdf).
    """

    def __init__(
        self,
        model_config: AttrDict,
        in_dim: int,
        out_dim: int,
        patch_out_dim: int = 0,
        norm: Optional[str] = None,
        act: str = "gelu",
        last_norm: Optional[str] = None,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        shared_head: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert (
            num_layers >= 2
        ), "iBOT distillation head number of layers should be at least 2"
        patch_out_dim = patch_out_dim or out_dim

        norm = self._build_norm(norm, hidden_dim)
        act = self._build_act(act)

        # Build the MLP
        layers = [nn.Linear(in_dim, hidden_dim)]
        if norm is not None:
            layers.append(norm)
        layers.append(act)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm is not None:
                layers.append(norm)
            layers.append(act)
        self.mlp = CustomSequential(*layers)

        # Build the projections
        self.to_class_out = nn.Linear(hidden_dim, out_dim)
        if not shared_head:
            self.to_patch_out = nn.Linear(hidden_dim, patch_out_dim)
        else:
            self.to_patch_out = self.to_class_out

        # Initialize the weights
        self.apply(self._init_weights)

        # Build the last norm for class token and patch token
        self.last_norm = self._build_norm(last_norm, out_dim, affine=False, **kwargs)
        if not shared_head:
            self.last_norm2 = self._build_norm(
                last_norm, patch_out_dim, affine=False, **kwargs
            )
        else:
            self.last_norm2 = self.last_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _build_norm(norm: Optional[str], hidden_dim: int, **kwargs):
        if norm == "bn":
            norm = nn.BatchNorm1d(hidden_dim, **kwargs)
        elif norm == "syncbn":
            norm = nn.SyncBatchNorm(hidden_dim, **kwargs)
        elif norm == "ln":
            norm = nn.LayerNorm(hidden_dim, **kwargs)
        else:
            assert norm is None, "unknown norm type {}".format(norm)
        return norm

    @staticmethod
    def _build_act(act: str):
        if act == "relu":
            act = nn.ReLU()
        elif act == "gelu":
            act = nn.GELU()
        else:
            raise ValueError("Unknown activation type {}".format(act))
        return act

    def forward(self, x):
        # Case of the local views:
        # - Only the class token is produced out of the trunk
        # - We run the equivalent of a DINO Head
        if len(x.shape) == 2:
            return self.forward_local_views(x)

        # Case of the global views:
        # - Class token + feature map are output by the trunk
        # - We compute the prototypes for all tokens
        else:
            return self.forward_global_views(x)

    def forward_local_views(self, x):
        x = self.mlp(x)
        x = self.to_class_out(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return [x]

    def forward_global_views(self, x):
        x = self.mlp(x)
        x1 = self.to_class_out(x[:, 0])
        x2 = self.to_patch_out(x[:, 1:])

        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)

        # Return the class embedding first, patch embedding second
        return [x1, x2]
