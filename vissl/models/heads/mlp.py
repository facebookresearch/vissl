# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn


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
        model_config,
        dims,
        use_bn=False,
        use_relu=False,
        use_dropout=False,
        use_bias=True,
    ):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            use_bn (bool): whether to attach BatchNorm after Linear layer
            use_relu (bool): whether to attach ReLU after (Linear (-> BN optional))
            use_dropout (bool): whether to attach Dropout after
                                (Linear (-> BN -> relu optional))
            use_bias (bool): whether the Linear layer should have bias or not
            dims (int): dimensions of the linear layer. Example [8192, 1000] which means
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

    def forward(self, batch):
        """
        Args:
            batch (torch.Tensor): 2D torch tensor or 4D tensor of shape `N x C x 1 x 1`
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        batch = torch.squeeze(batch)
        out = self.clf(batch)
        return out
