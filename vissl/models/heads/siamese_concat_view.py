# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
from vissl.models.heads import register_model_head
from vissl.utils.hydra_config import AttrDict


@register_model_head("siamese_concat_view")
class SiameseConcatView(nn.Module):
    """
    This head is useful for dealing with Siamese models which have multiple towers.
    For an input of type (N * num_towers) x C, this head can convert the output
    to N x (num_towers * C).

    This head is used in case of PIRL https://arxiv.org/abs/1912.01991 and
    Jigsaw https://arxiv.org/abs/1603.09246 approaches.
    """

    def __init__(self, model_config: AttrDict, num_towers: int):
        """
        Args:
            model_config (AttrDict): dictionary config.MODEL in the config file

            num_towers (int): number of towers in siamese model
        """
        super().__init__()
        self.num_towers = num_towers

    def forward(self, batch: torch.Tensor):
        """
        Args:
            batch (torch.Tensor): 2D torch tensor `(N * num_towers) x C` or 4D tensor of
                                  shape `(N * num_towers) x C x 1 x 1`
        Returns:
            out (torch.Tensor): 2D output torch tensor `N x (C * num_towers)`
        """
        # batch dimension = (N * num_towers) x C x H x W
        siamese_batch_size = batch.shape[0]
        assert (
            siamese_batch_size % self.num_towers == 0
        ), f"{siamese_batch_size} not divisible by num_towers {self.num_towers}"
        batch_size = siamese_batch_size // self.num_towers
        out = batch.view(batch_size, -1)
        return out
