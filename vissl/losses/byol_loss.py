# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging
import pprint
from collections import namedtuple

import torch
import torch.nn.functional as F
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.utils.misc import concat_all_gather


_BYOLLossConfig = namedtuple(
    "_BYOLLossConfig", ["embedding_dim", "momentum"]
)

def regression_loss(x, y):
    normed_x, normed_y = F.normalize(x, dim=1), F.normalize(y, dim=1)
    # Euclidean Distance squared.
    return 2 - 2 * (normed_x * normed_y).sum(dim=1)


class BYOLLossConfig(_BYOLLossConfig):
    """ Settings for the BYOL loss"""

    @staticmethod
    def defaults() -> "BYOLLossConfig":
        return BYOLLossConfig(
            embedding_dim=256, momentum=0.999
        )


@register_loss("byol_loss")
class BYOLLoss(ClassyLoss):
    """
    TODO: change description

    This is the loss which was proposed in the "Momentum Contrast
    for Unsupervised Visual Representation Learning" paper, from Kaiming He et al.
    See http://arxiv.org/abs/1911.05722 for details
    and https://github.com/facebookresearch/moco for a reference implementation, reused here

    Config params:
        embedding_dim (int): head output output dimension
        momentum (float): encoder momentum value for the update
    """

    def __init__(self, config: BYOLLossConfig):
        super().__init__()
        self.loss_config = config
        self.target_network = None
        self.checkpoint = None

    @classmethod
    def from_config(cls, config: BYOLLossConfig):
        """
        Instantiates BYOLLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            BYOLLoss instance.
        """
        return cls(config)

    def forward(self, online_network_prediction: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Given the encoder queries, the key and the queue of the previous queries,
        compute the cross entropy loss for this batch

        Args:
            online_network_prediction: online model output. this is a prediction of the
            target network output.

        Returns:
            loss
        """

        # Split data
        online_view1, online_view2 = torch.chunk(online_network_prediction, 2, 0)
        target_view1, target_view2 = torch.chunk(self.target_embs.detach(), 2, 0)

        # TESTED: Views are received correctly.

        # Compute losses
        loss1 = regression_loss(online_view1, target_view2)
        loss2 = regression_loss(online_view2, target_view1)
        loss = (loss1 + loss2).mean()

        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Restore the loss state given a checkpoint

        Args:
            state_dict (serialized via torch.save)
        """
        # If the encoder has been allocated, use the normal pytorch restoration
        if self.target_network is None:
            self.checkpoint = state_dict
            logging.info("Storing the checkpoint for later use")
        else:
            logging.info("Restoring checkpoint")
            super().load_state_dict(state_dict, *args, **kwargs)
