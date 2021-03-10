# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pprint

import torch
from classy_vision.generic.distributed_util import get_cuda_device_index, get_rank
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.utils.hydra_config import AttrDict


@register_loss("barlow_twins_loss")
class BarlowTwinsLoss(ClassyLoss):
    """
    This is the loss which was proposed in the Barlow Twins https://arxiv.org/abs/2103.03230v1
    paper.
    See the paper for the details on the loss.

    Config params:
        lambda_ (float): weight on the off-diagonal terms. It controls the tradeoffs between
            the importance given to the invariance term versus the redundancy reduction term.
        embedding_dim (int): dimensionality of the representation
    """

    def __init__(self, loss_config: AttrDict):
        super(BarlowTwinsLoss, self).__init__()

        self.loss_config = loss_config
        self.bt_criterion = BarlowTwinsCriterion(
            self.loss_config.lambda_, self.loss_config.embedding_dim
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates BarlowTwinsLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            BarlowTwinsLoss instance.
        """
        return cls(loss_config)

    def forward(self, output, target):
        loss = self.bt_criterion(output)
        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "lambda_": self.loss_config.lambda_,
            "embedding_dim": self.loss_config.embedding_dim
        }
        return pprint.pformat(repr_dict, indent=2)


class BarlowTwinsCriterion(nn.Module):
    """
    The criterion corresponding to the Barlow Twins loss as defined in the paper
    https://arxiv.org/abs/2103.03230v1.

    Args:
        lambda_ (float): weight on the off-diagonal terms. It controls the tradeoffs between
            the importance given to the invariance term versus the redundancy reduction term.
        embedding_dim (int): dimensionality of the representation
    """

    def __init__(self, lambda_: float, embedding_dim: int):
        super(BarlowTwinsCriterion, self).__init__()

        self.use_gpu = get_cuda_device_index() > -1
        self.lambda_ = lambda_
        self.embedding_dim = embedding_dim
        self.num_copies = 2

        identity_matrix = torch.eye(embedding_dim)
        if self.use_gpu:
            identity_matrix = identity_matrix.cuda(non_blocking=True)

        self.identity_matrix = identity_matrix

    def forward(self, embedding: torch.Tensor):
        """
        Calculate the loss. Operates on embeddings tensor.
        """
        assert embedding.ndim == 2

        batch_size = embedding.shape[0]
        L = self.lambda_
        num_copies = self.num_copies
        assert batch_size % num_copies == 0, "Batch size should be divisible by num_pos"

        # normalize embeddings along the batch dimension
        embedding_normed = (embedding - embedding.mean(dim=0)) / (embedding.std(dim=0) + 1e-16)
        embedding_normed_a, embedding_normed_b = torch.split(embedding_normed,
                                                             split_size_or_sections=batch_size // num_copies,
                                                             dim=0)

        # cross-correlation matrix
        c = torch.mm(embedding_normed_a.T, embedding_normed_b) / batch_size

        # loss
        c_diff = (c - self.identity_matrix).pow(2)

        # multiply off-diagonal elements of c_diff by lambda
        c_diff[~self.identity_matrix.bool()] *= L
        loss = c_diff.sum()

        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "lambda_": self.lambda_,
            "embedding_dim": self.embedding_dim,
            "num_copies": self.num_copies
        }
        return pprint.pformat(repr_dict, indent=2)
