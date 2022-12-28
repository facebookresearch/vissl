# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.nn as nn
from vissl.config import AttrDict
from vissl.losses.cross_entropy_multiple_output_single_target import EnsembleOutput
from vissl.models.heads import register_model_head
from vissl.models.heads.mlp import MLP


@register_model_head("eval_mlp_pooled")
class LinearEvalPooledMLP(nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        model_config: AttrDict,
        in_channels: int,
        dims: List[int],
        use_bn: bool = False,
        use_relu: bool = False,
    ):
        super().__init__()
        # '''
        self.channel_bn = nn.BatchNorm2d(
            in_channels,
            eps=model_config.HEAD.BATCHNORM_EPS,
            momentum=model_config.HEAD.BATCHNORM_MOMENTUM,
        )
        # '''
        # self.channel_bn = nn.Identity()
        layers = MLP.create_layers(model_config, dims, use_bn=use_bn, use_relu=use_relu)
        self.clf = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 3D torch tensor.
        Returns:
            out (torch.Tensor): 2D output torch tensor
        """
        assert len(batch.shape) == 3, "Expecting shape (batch, seq, embed)"
        batch = batch.permute((0, 2, 1)).unsqueeze(3)

        assert len(batch.shape) == 4, "Expecting shape (batch, embed, seq, 1)"
        out = self.channel_bn(batch)
        out = torch.flatten(out, start_dim=2)

        # Reshape: batch, embed, seq -> seq, batch, embed
        out = out.permute((2, 0, 1))
        out = self.clf(out)
        return EnsembleOutput(out)
