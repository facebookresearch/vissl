# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.models.model_helpers import trunc_normal_


class CustomSequential(nn.Sequential):
    """Custom sequential head that permute dimensions for variants of BN
    Credits goes to iBOT codebase:
    https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/models/head.py#L51
    """

    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self:
            dim = len(x.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1))
                perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]
                inv_perm.pop(1)
                x = module(x.permute(*perm)).permute(*inv_perm)
            else:
                x = module(x)
        return x


@register_model_head("ibot_head")
class IBOTHead(nn.Module):
    """
    Specific head for training iBOT (https://arxiv.org/pdf/2111.07832.pdf).

    Adapted from the official iBOT code base to fit in VISSL:
    https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/models/head.py#L145
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
        normalize_last_layer: bool = True,
        num_layers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        shared_head: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert num_layers >= 1, "iBOT head number of layers should be at least 1"
        patch_out_dim = patch_out_dim or out_dim

        # TODO(IBOT) - bug in the official repo? each norm should be a
        #  separate instance (not used several times inside Sequential)
        norm = self._build_norm(norm, hidden_dim)
        act = self._build_act(act)

        # Build the MLP
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if norm is not None:
                    layers.append(norm)
                layers.append(act)
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = CustomSequential(*layers)
        self.apply(self._init_weights)

        # Build the prototypes for class token
        self.prototypes0 = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.prototypes0.weight_g.data.fill_(1)
        if normalize_last_layer:
            self.prototypes0.weight_g.requires_grad = False

        # Build the prototypes for patch tokens
        if not shared_head:
            self.prototypes1 = nn.utils.weight_norm(
                nn.Linear(bottleneck_dim, patch_out_dim, bias=False)
            )
            self.prototypes1.weight_g.data.fill_(1)
            if normalize_last_layer:
                self.prototypes1.weight_g.requires_grad = False
        else:
            self.prototypes1 = self.prototypes0

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
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.prototypes0(x)
        if self.last_norm is not None:
            x = self.last_norm(x)
        return [x]

    def forward_global_views(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x1 = self.prototypes0(x[:, 0])
        x2 = self.prototypes1(x[:, 1:])

        if self.last_norm is not None:
            x1 = self.last_norm(x1)
            x2 = self.last_norm2(x2)

        # Return the class embedding first, patch embedding second
        return [x1, x2]
