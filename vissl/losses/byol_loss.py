# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import pprint
from collections import namedtuple

import torch
import torch.nn.functional as F
from classy_vision.losses import ClassyLoss, register_loss


_BYOLLossConfig = namedtuple("_BYOLLossConfig", ["embedding_dim", "momentum"])


def regression_loss(x, y):
    """
    This function is used for computing loss between the prediction
    from the Online network and projection from the target network.
    This is simply the euclidean distance squared.
    """
    normed_x, normed_y = F.normalize(x, dim=1), F.normalize(y, dim=1)
    # Euclidean Distance squared.
    return 2 - 2 * (normed_x * normed_y).sum(dim=1)


class BYOLLossConfig(_BYOLLossConfig):
    """Settings for the BYOL loss"""

    @staticmethod
    def defaults() -> "BYOLLossConfig":
        return BYOLLossConfig(embedding_dim=256, momentum=0.999)


@register_loss("byol_loss")
class BYOLLoss(ClassyLoss):
    """
    This is the loss proposed in BYOL
    - Bootstrap your own latent: (https://arxiv.org/abs/2006.07733)
    This class wraps functions which computes
    - loss : BYOL uses contrastive loss which is the difference in
            l2-normalized Online network's prediction and Target
            network's projections or cosine similarity between the two.
            In this implementation we have used Cosine similarity.
    - restores loss from checkpoints.

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
    def from_config(cls, config: BYOLLossConfig) -> "BYOLLoss":
        """
        Instantiates BYOLLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            BYOLLoss instance.
        """
        return cls(config)

    def forward(
        self, online_network_prediction: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        In this function, the Online Network receives the tensor as input after projection
        and they make predictions on the output of the target network’s projection,
        The similarity between the two is computed and then a mean of it is used to
        update the parameters of both the networks to reduce loss.

        Given the encoder queries, the key and the queue of the previous queries,
        compute the cross entropy loss for this batch.

        Args:
            online_network_prediction: online model output. this is a prediction of the
            target network output.

        Returns:
            loss
        """

        # Split data
        online_view1, online_view2 = torch.chunk(online_network_prediction, 2, 0)
        target_view1, target_view2 = torch.chunk(self.target_embs.detach(), 2, 0)

        # Compute losses
        loss1 = regression_loss(online_view1, target_view2)
        loss2 = regression_loss(online_view2, target_view1)

        loss = (loss1 + loss2).mean()

        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)

    def load_state_dict(self, state_dict, *args, **kwargs) -> None:
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
