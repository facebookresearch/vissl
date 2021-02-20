# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import torch
import torch.nn as nn
from vissl.models.heads import register_model_head
from vissl.models.heads.mlp import MLP
from vissl.utils.hydra_config import AttrDict


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

    def __init__(
        self,
        model_config: AttrDict,
        in_channels: int,
        dims: List[int],
        use_bn: bool = False,
        use_relu: bool = False,
    ):
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

        self.clf = MLP(model_config, dims, use_bn=use_bn, use_relu=use_relu)

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 4D torch tensor. This layer is meant to be attached at several
                                  parts of the model to evaluate feature representation quality.
                                  For 2D input tensor, the tensor is unsqueezed to NxDx1x1 and
                                  then eval_mlp is applied
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        # in case of a 2D tensor input, unsqueeze the tensor to (N x D x 1 x 1) and apply
        # eval mlp normally.
        if len(batch.shape) == 2:
            batch = batch.unsqueeze(2).unsqueeze(3)
        assert len(batch.shape) == 4, "Eval MLP head expects 4D tensor input"
        out = self.channel_bn(batch)
        out = torch.flatten(out, start_dim=1)
        out = self.clf(out)
        return out
