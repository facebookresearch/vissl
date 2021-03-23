# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pprint

import torch
from classy_vision.generic.distributed_util import all_reduce_mean, get_world_size
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from torch.autograd.function import Function
from vissl.utils.distributed_gradients import gather_from_all
from vissl.utils.hydra_config import AttrDict


class SyncNormalizeFunction(Function):
    """
    Adapted from: https://github.com/NVIDIA/apex/blob/master/apex/parallel/sync_batchnorm.py

    Normalizes a NxD input over the first dimension and across all processes.
    """
    @staticmethod
    def forward(ctx, input, eps):
        with torch.no_grad():
            local_mean = torch.mean(input, 0)
            local_sqr_mean = torch.pow(input, 2).mean(0)

            # If running on a distributed setting, perform mean reduction of tensors over
            # all processes.
            mean = all_reduce_mean(local_mean)
            sqr_mean = all_reduce_mean(local_sqr_mean)

            # var(x) = E (( x - mean_x ) ** 2)
            #        = 1 / N * sum ( x - mean_x ) ** 2
            #        = 1 / N * sum (x**2) - mean_x**2
            var = sqr_mean - mean.pow(2)

        # transpose it to channel last to support broadcasting for input with different rank
        c_last_input = input.transpose(1, -1).contiguous().clone()

        ctx.save_for_backward(c_last_input, mean, var)
        ctx.eps = eps

        c_last_input = (c_last_input - mean) / torch.sqrt(var + eps)

        return c_last_input.transpose(1, -1).contiguous().clone()

    @staticmethod
    def backward(ctx, grad_output):
        # mini batch mean & var are calculated by forward path.
        # mu = 1./N*np.sum(h, axis = 0)
        # var = 1./N*np.sum((h-mu)**2, axis = 0)
        c_last_input, mean, var = ctx.saved_tensors

        eps = ctx.eps
        grad_input = None
        num_features = mean.size()[0]

        # calculate grad_input
        if ctx.needs_input_grad[0]:
            # dh = gamma * (var + eps)**(-1. / 2.) * (dy - np.mean(dy, axis=0)
            #     - (h - mu) * (var + eps)**(-1.0) * np.mean(dy * (h - mu), axis=0))
            mean_dy = grad_output.mean(0)
            mean_dy_xmu = (grad_output * (c_last_input -
                                          mean)).view(-1, num_features).mean(0)
            # If running on a distributed setting, perform mean reduction of tensors over
            # all processes.
            mean_dy = all_reduce_mean(mean_dy)
            mean_dy_xmu = all_reduce_mean(mean_dy_xmu)

            grad_input = (grad_output - mean_dy - (c_last_input - mean) / (
                    var + eps) * mean_dy_xmu) / torch.sqrt(var + eps)

        return grad_input, None


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

        self.lambda_ = lambda_
        self.scale_loss = scale_loss
        self.embedding_dim = embedding_dim
        self.num_copies = 2
        self.eps = 1e-5

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """
        return a flattened view of the off-diagonal elements of a square matrix
       """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

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
        embedding_normed = SyncNormalizeFunction.apply(embedding, self.eps)

        # split embedding between copies
        embedding_normed_a, embedding_normed_b = torch.split(
            embedding_normed,
            split_size_or_sections=batch_size // self.num_copies,
            dim=0,
        )

        # Do a gather over all embeddings, so we can compute the loss.
        # Final shape is: (batch_size * num_gpus) x embedding_dim
        embedding_normed_a = gather_from_all(embedding_normed_a)
        embedding_normed_b = gather_from_all(embedding_normed_b)

        # cross-correlation matrix
        c = torch.mm(embedding_normed_a.T, embedding_normed_b) / (
                batch_size * get_world_size())

        # loss
        on_diag = torch.diagonal(c).add(-1).pow(2).sum().mul(self.scale_loss)
        off_diag = self._off_diagonal(c).pow(2).sum().mul(self.scale_loss)
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
