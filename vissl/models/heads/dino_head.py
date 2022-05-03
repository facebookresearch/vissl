# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.models.model_helpers import trunc_normal_
from vissl.utils.fsdp_utils import fsdp_wrapper


class NormalizedLinearLayer(nn.Module):
    """
    Linear layer where the weights are normalized, equivalent to the output
    of "nn.utils.weight_norm" but compatible with FSDP
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_weight_v: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.weight_v = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_g", torch.Tensor(out_features, 1))
        if init_weight_v is not None:
            self.weight_v.data.copy_(init_weight_v)
        self.weight_g.data.fill_(1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # The following code is equivalent to, but keeps more
        # precision than:
        #
        #   norm = self.weight_v.norm(dim=-1, keepdim=True)
        #   weight = self.weight_v / norm * self.weight_g
        #
        # And wo we keep it that way:
        weight_norm = WeightNorm(name="weight", dim=0)
        weight = weight_norm.compute_weight(self)
        return F.linear(input, weight)


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


@register_model_head("dino_head_fsdp")
def DINOHeadFSDP(
    model_config: AttrDict,
    in_dim: int,
    num_clusters: List[int],
    use_bn: bool = False,
    normalize_last_layer: bool = True,
    num_layers: int = 3,
    hidden_dim: int = 2048,
    bottleneck_dim: int = 256,
):
    head = DINOHead(
        model_config=model_config,
        in_dim=in_dim,
        num_clusters=num_clusters,
        use_bn=use_bn,
        normalize_last_layer=normalize_last_layer,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
    )

    # TODO - FSDP does not work around "nn.utils.weight_norm" when
    #  required_grad is set to False on "weight_g", as it seems that
    #  FSDP requires all the parameters to be requiring grad or none
    #  of them, so we workaround it by making "weight_g" a buffer
    #  instead of a nn.Parameter
    if normalize_last_layer:
        head.prototypes0 = NormalizedLinearLayer(
            bottleneck_dim,
            num_clusters[0],
            init_weight_v=head.prototypes0.weight_v.data,
        )

    # Wrap prototypes in FP32
    prototypes_fp32_fsdp_config = model_config.FSDP_CONFIG.copy()
    prototypes_fp32_fsdp_config["flatten_parameters"] = False
    prototypes_fp32_fsdp_config["mixed_precision"] = False
    prototypes_fp32_fsdp_config["fp32_reduce_scatter"] = False
    prototypes_fp32_fsdp_config["compute_dtype"] = torch.float32
    head.prototypes0 = fsdp_wrapper(head.prototypes0, **prototypes_fp32_fsdp_config)

    # Wrap the rest of the head
    return fsdp_wrapper(head, **model_config.FSDP_CONFIG)
