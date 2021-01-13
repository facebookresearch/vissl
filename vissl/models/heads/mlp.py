# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
import torch.nn as nn
from vissl.models.heads import register_model_head
from vissl.utils.hydra_config import AttrDict


@register_model_head("mlp")
class MLP(nn.Module):
    """
    This module can be used to attach combination of {Linear, BatchNorm, Relu, Dropout}
    layers and they are fully configurable from the config file. The module also supports
    stacking multiple MLPs.

    Examples:
        Linear
        Linear -> BN
        Linear -> ReLU
        Linear -> Dropout
        Linear -> BN -> ReLU -> Dropout
        Linear -> ReLU -> Dropout
        Linear -> ReLU -> Linear -> ReLU -> ...
        Linear -> Linear -> ...
        ...

    Accepts a 2D input tensor. Also accepts 4D input tensor of shape `N x C x 1 x 1`.
    """

    def __init__(
        self,
        model_config: AttrDict,
        dims: List[int],
        use_bn: bool = False,
        use_relu: bool = False,
        use_dropout: bool = False,
        use_bias: bool = True,
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            use_bn (bool): whether to attach BatchNorm after Linear layer
            use_relu (bool): whether to attach ReLU after (Linear (-> BN optional))
            use_dropout (bool): whether to attach Dropout after
                                (Linear (-> BN -> relu optional))
            use_bias (bool): whether the Linear layer should have bias or not
            dims (int): dimensions of the linear layer. Example [8192, 1000] which
                        attaches `nn.Linear(8192, 1000, bias=True)`
        """
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
        # we use the default normal or uniform initialization for the layers
        # and allow users to scale the initialization.
        self.scale_weights(model_config)

    def scale_weights(self, model_config):
        params_multiplier = model_config.HEAD.PARAMS_MULTIPLIER
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data *= params_multiplier
                if m.bias is not None:
                    m.bias.data *= params_multiplier

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
            ), "MLP input should be either a tensor (2D, 4D) or list containing 1 tensor."
            batch = batch[0]
        batch = torch.squeeze(batch)
        assert (
            len(batch.shape) <= 2
        ), f"MLP expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
        out = self.clf(batch)
        return out
