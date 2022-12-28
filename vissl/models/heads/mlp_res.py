# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head


class MLPResidualBlock(nn.Module):
    def __init__(
        self,
        model_config: AttrDict,
        input_dim: int,
        output_dim: int,
        use_bias: bool,
        use_bn: bool,
        use_relu: bool,
        use_dropout: bool,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
        if use_bn:
            layers.append(
                nn.BatchNorm1d(
                    output_dim,
                    eps=model_config.HEAD.BATCHNORM_EPS,
                    momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
                )
            )
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        if use_dropout:
            layers.append(nn.Dropout())
        self.mlp_res_block = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_dim == self.output_dim:
            return x + self.mlp_res_block(x)
        else:
            return self.mlp_res_block(x)


@register_model_head("mlp_res")
class MLPResidual(nn.Module):
    """
    A variant of MLP in which we introduce skip connections between
    successive layers in case the dimensions allow it
    """

    def __init__(
        self,
        model_config: AttrDict,
        dims: List[int],
        use_bn: bool = False,
        use_relu: bool = False,
        use_dropout: bool = False,
        use_bias: bool = True,
        skip_last_layer_relu_bn: bool = True,
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
            skip_last_layer_relu_bn (bool): If the MLP has many layers, we check
                if after the last MLP layer, we should add BN / ReLU or not. By
                default, skip it. If user specifies to not skip, then BN will be
                added if use_bn=True, ReLU will be added if use_relu=True
        """
        super().__init__()
        layers = self.create_layers(
            model_config,
            dims,
            use_bn,
            use_relu,
            use_dropout,
            use_bias,
            skip_last_layer_relu_bn,
        )
        self.clf = nn.Sequential(*layers)
        # we use the default normal or uniform initialization for the layers
        # and allow users to scale the initialization.
        self.scale_weights(model_config)

    @staticmethod
    def create_layers(
        model_config: AttrDict,
        dims: List[int],
        use_bn: bool = False,
        use_relu: bool = False,
        use_dropout: bool = False,
        use_bias: bool = True,
        skip_last_layer_relu_bn: bool = True,
    ):
        layers = []
        last_dim = dims[0]
        for i, dim in enumerate(dims[1:]):

            keep_relu_bn_drop = not (i == len(dims) - 2 and skip_last_layer_relu_bn)
            layers.append(
                MLPResidualBlock(
                    model_config=model_config,
                    input_dim=last_dim,
                    output_dim=dim,
                    use_bias=use_bias,
                    use_bn=use_bn and keep_relu_bn_drop,
                    use_relu=use_relu and keep_relu_bn_drop,
                    use_dropout=use_dropout and keep_relu_bn_drop,
                )
            )
            last_dim = dim
        return layers

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

        if batch.ndim > 2:
            assert all(
                d == 1 for d in batch.shape[2:]
            ), f"MLP expected 2D input tensor or 4D tensor of shape NxCx1x1. got: {batch.shape}"
            batch = batch.reshape((batch.size(0), batch.size(1)))

        x = batch
        for module in self.clf:
            delta = module(x)
            if x.shape == delta.shape:
                x = delta + x
            else:
                x = delta
        return x
