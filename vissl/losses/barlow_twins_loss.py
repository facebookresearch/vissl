# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pprint

import torch
from classy_vision.generic.distributed_util import (
    all_reduce_mean,
    get_cuda_device_index,
)
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
        lambda_ (float):        weight on the off-diagonal terms. It controls the trade-off
                                between the importance given to the invariance term versus the
                                redundancy reduction term.
        scale_loss (float):     In order to match the code that was used to develop Barlow
                                Twins, we include an additional parameter, scale-loss, that
                                multiplies the loss by a constant factor. We are working on a
                                version that will not require this parameter.
        embedding_dim (int):    dimensionality of the representation
    """

    def __init__(self, loss_config: AttrDict):
        super(BarlowTwinsLoss, self).__init__()

        self.loss_config = loss_config
        self.bt_criterion = BarlowTwinsCriterion(
            lambda_=self.loss_config.lambda_,
            scale_loss=self.loss_config.scale_loss,
            embedding_dim=self.loss_config.embedding_dim,
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict) -> "BarlowTwinsLoss":
        """
        Instantiates BarlowTwinsLoss from configuration.

        Args:
            loss_config:    configuration for the loss

        Returns:
            BarlowTwinsLoss instance.
        """
        return cls(loss_config)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bt_criterion(output)
        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "lambda_": self.loss_config.lambda_,
            "scale_loss": self.loss_config.scale_loss,
            "embedding_dim": self.loss_config.embedding_dim,
        }
        return pprint.pformat(repr_dict, indent=2)


class BarlowTwinsCriterion(nn.Module):
    """
    The criterion corresponding to the Barlow Twins loss as defined in the paper
    https://arxiv.org/abs/2103.03230v1.

    Args:
        lambda_ (float):        weight on the off-diagonal terms. It controls the trade-off
                                between the importance given to the invariance term versus the
                                redundancy reduction term.
        scale_loss (float):     In order to match the code that was used to develop Barlow
                                Twins, we include an additional parameter, scale-loss, that
                                multiplies the loss by a constant factor. We are working on a
                                version that will not require this parameter.
        embedding_dim (int):    dimensionality of the representation
    """

    def __init__(self, lambda_: float, scale_loss: float, embedding_dim: int):
        super(BarlowTwinsCriterion, self).__init__()

        self.use_gpu = get_cuda_device_index() > -1
        self.lambda_ = lambda_
        self.scale_loss = scale_loss
        self.embedding_dim = embedding_dim
        self.num_copies = 2
        self.eps = 1e-5

        self.diag_mask = torch.eye(embedding_dim, dtype=torch.bool)

    def _sync_normalize(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Adapted from: https://github.com/NVIDIA/apex/blob/master/apex/parallel/sync_batchnorm.py
        """
        with torch.no_grad():
            local_mean = torch.mean(embedding, 0)
            local_sqr_mean = torch.pow(embedding, 2).mean(0)

            # If running on a distributed setting, perform mean reduction of tensors over
            # all processes.
            mean = all_reduce_mean(local_mean)
            sqr_mean = all_reduce_mean(local_sqr_mean)

            # var(x) = E (( x - mean_x ) ** 2)
            #        = 1 / N * sum ( x - mean_x ) ** 2
            #        = 1 / N * sum (x**2) - mean_x**2
            var = sqr_mean - mean.pow(2)

        return (embedding - mean) / torch.sqrt(var + self.eps)

    def forward(self, embedding: torch.Tensor) -> torch.tensor:
        """
        Calculate the loss. Operates on embeddings tensor.

        Args:
            embedding (torch.Tensor):   NxEMBEDDING_DIM
                                        Must contain the concatenated embeddings
                                        of the two image copies:
                                        [emb_img1_0, emb_img2_0, ....., emb_img1_1, emb_img2_1,...]
        """
        assert embedding.ndim == 2 and embedding.shape[1] == int(
            self.embedding_dim
        ), f"Incorrect embedding shape: {embedding.shape} but expected Nx{self.embedding_dim}"

        batch_size = embedding.shape[0]
        assert (
            batch_size % self.num_copies == 0
        ), f"Batch size {batch_size} should be divisible by num_copies ({self.num_copies})."

        # normalize embeddings along the batch dimension
        embedding_normed = self._sync_normalize(embedding)

        # split embedding between copies
        embedding_normed_a, embedding_normed_b = torch.split(
            embedding_normed,
            split_size_or_sections=batch_size // self.num_copies,
            dim=0,
        )

        # cross-correlation matrix
        c = torch.mm(embedding_normed_a.T, embedding_normed_b) / batch_size

        # If running on a distributed setting, perform mean reduction of c over
        # all processes.
        c = all_reduce_mean(c)

        # loss
        on_diag = c[self.diag_mask].add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = c[~self.diag_mask].pow_(2).sum().mul(self.scale_loss)
        loss = on_diag + self.lambda_ * off_diag

        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "lambda_": self.lambda_,
            "scale_loss": self.scale_loss,
            "embedding_dim": self.embedding_dim,
            "num_copies": self.num_copies,
        }
        return pprint.pformat(repr_dict, indent=2)
