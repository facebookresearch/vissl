# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.utils.fsdp_utils import fsdp_auto_wrap_bn, fsdp_wrapper


@register_model_head("swav_head")
class SwAVPrototypesHead(nn.Module):
    """
    SwAV head used in https://arxiv.org/pdf/2006.09882.pdf paper.

    The head is composed of 2 parts
        1) projection of features to lower dimension like 128
        2) feature classification into clusters (also called prototypes)

    The projected features are L2 normalized before clustering step.

    Input: 2D torch.tensor of shape (N x C)

    Output: List(2D torch.tensor of shape N x num_clusters)
    """

    def __init__(
        self,
        model_config: AttrDict,
        dims: List[int],
        use_bn: bool,
        num_clusters: int,
        use_bias: bool = True,
        return_embeddings: bool = True,
        skip_last_bn: bool = True,
        normalize_feats: bool = True,
        activation_name: str = "ReLU",
        use_weight_norm_prototypes: bool = False,
        normalize_last_layer: bool = False,
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            dims (int): dimensions of the linear layer. Must have length at least 2.
                        Example: [2048, 2048, 128] attaches linear layer
                                 Linear(2048, 2048) -> BN -> Relu -> Linear(2048, 128)
            use_bn (bool): whether to attach BatchNorm after Linear layer
            num_clusters (List(int)): number of prototypes or clusters. Typically 3000.
                                      Example dims=[3000] will attach 1 prototype head.
                                              dims=[3000, 3000] will attach 2 prototype heads
            use_bias (bool): whether the Linear layer should have bias or not
            return_embeddings (bool): whether return the projected embeddings or not
            skip_last_bn (bool): whether to attach BN + Relu at the end of projection head.
                        Example:
                            [2048, 2048, 128] with skip_last_bn=True attaches linear layer
                            Linear(2048, 2048) -> BN -> Relu -> Linear(2048, 128)

                            [2048, 2048, 128] with skip_last_bn=False attaches linear layer
                            Linear(2048, 2048) -> BN -> Relu -> Linear(2048, 128) -> BN -> ReLU

                        This could be particularly useful when performing full finetuning on
                        hidden layers.
            use_weight_norm_prototypes (bool): whether to use weight norm module for the
            prototypes layers.
        """

        super().__init__()
        self.normalize_feats = normalize_feats
        # build the projection head
        layers = []
        last_dim = dims[0]
        for i, dim in enumerate(dims[1:]):
            layers.append(nn.Linear(last_dim, dim, bias=use_bias))
            if (i == len(dims) - 2) and skip_last_bn:
                break
            if use_bn:
                layers.append(
                    nn.BatchNorm1d(
                        dim,
                        eps=model_config.HEAD.BATCHNORM_EPS,
                        momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
                    )
                )
            if activation_name == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            if activation_name == "GELU":
                layers.append(nn.GELU())
            last_dim = dim
        self.projection_head = nn.Sequential(*layers)

        # prototypes (i.e. centroids) layers
        if len(num_clusters) > 0:
            self.nmb_heads = len(num_clusters)
            for i, k in enumerate(num_clusters):
                proto = nn.Linear(dims[-1], k, bias=False)
                if use_weight_norm_prototypes:
                    proto = nn.utils.weight_norm(proto)
                    proto.weight_g.data.fill_(1)
                self.add_module("prototypes" + str(i), proto)
                if normalize_last_layer:
                    proto.weight_g.requires_grad = False
        else:
            self.nmb_heads = 0
        self.return_embeddings = return_embeddings

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (4D torch.tensor): shape (N x C x H x W)
        Returns:
            List(2D torch.tensor of shape N x num_clusters)
        """
        batch = self.projection_head(batch)

        if self.normalize_feats:
            batch = nn.functional.normalize(batch, dim=1, p=2)

        out = []
        if self.return_embeddings:
            out.append(batch)
        if self.nmb_heads > 0:
            for i in range(self.nmb_heads):
                out.append(getattr(self, "prototypes" + str(i))(batch))
        return out


@register_model_head("swav_head_fsdp")
def SwavPrototypesHeadFSDP(
    model_config: AttrDict,
    dims: List[int],
    use_bn: bool,
    num_clusters: int,
    use_bias: bool = True,
    return_embeddings: bool = True,
    skip_last_bn: bool = True,
    normalize_feats: bool = True,
):
    """
    SwAV head specific FSDP wrapping: we keep the full precision for the prototypes

    This is important for convergence:
    Since we "normalize" this layer in the update hook, we need to keep its
    weights in full precision. It is output is going into the loss and used
    for clustering, so we need to have that in full precision as well.
    """

    head = SwAVPrototypesHead(
        model_config=model_config,
        dims=dims,
        use_bn=use_bn,
        num_clusters=num_clusters,
        use_bias=use_bias,
        return_embeddings=return_embeddings,
        skip_last_bn=skip_last_bn,
        normalize_feats=normalize_feats,
    )
    head = fsdp_auto_wrap_bn(head)

    prototypes_fp32_fsdp_config = model_config.FSDP_CONFIG.copy()
    prototypes_fp32_fsdp_config["flatten_parameters"] = False
    prototypes_fp32_fsdp_config["mixed_precision"] = False
    prototypes_fp32_fsdp_config["fp32_reduce_scatter"] = False
    prototypes_fp32_fsdp_config["compute_dtype"] = torch.float32
    for j in range(head.nmb_heads):
        module = getattr(head, "prototypes" + str(j))
        module = fsdp_wrapper(module, **prototypes_fp32_fsdp_config)
        setattr(head, "prototypes" + str(j), module)

    return fsdp_wrapper(head, **model_config.FSDP_CONFIG)
