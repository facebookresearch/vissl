# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
from vissl.models.heads import register_model_head
from vissl.models.heads.mlp import MLP


@register_model_head("eval_mlp")
class LinearEvalMLP(nn.Module):
    """
    A standard Linear classification module that can be attached to several
    layers of the model to evaluate the representation quality of features.

     The layers attached are:
        BatchNorm2d -> Linear (1 or more)

    Accepts a 4D input tensor. If you want to use 2D input tensor instead,
    use the "mlp" head directly.
    """

    def __init__(self, model_config, in_channels, dims):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file
            in_channels (int): number of channels the input has. This information is
                               used to attached the BatchNorm2D layer.
            dims (int): dimensions of the linear layer. Example [8192, 1000] which means
                        attaches `nn.Linear(8192, 1000, bias=True)`
        """

        super().__init__()
        self.channel_bn = nn.BatchNorm2d(
            in_channels,
            eps=model_config.HEAD.BATCHNORM_EPS,
            momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
        )

        self.clf = MLP(model_config, dims)

    def forward(self, batch):
        """
        Args:
            batch (torch.Tensor): 4D torch tensor. This layer is meant to be attached at several
                                  parts of the model to evaluate feature representation quality
                                  for 2D input tensor, see "mlp" head.
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        out = self.channel_bn(batch)
        out = torch.flatten(out, start_dim=1)
        out = self.clf(out)
        return out
