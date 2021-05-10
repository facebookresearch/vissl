# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import pprint
from typing import Dict

import torch
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict
from vissl.losses.simclr_info_nce_loss import SimclrInfoNCECriterion
from vissl.utils.misc import concat_all_gather


@register_loss("nnclr_info_nce_loss")
class NNclrInfoNCELoss(ClassyLoss):
    """
    This is the loss which was proposed in the NNCLR https://arxiv.org/abs/2104.14548 paper.
    See the paper for the details on the loss.

    Config params:
        prediction_head_params (dict): prediction MLP parameters
        temperature (float): the temperature to be applied on the logits
        queue_size (int): size of the nearest neighbor support set
        buffer_params:
            world_size (int): total number of trainers in training
            embedding_dim (int): output dimensions of the features projects
            effective_batch_size (int): total batch size used (includes positives)
    """

    def __init__(self, loss_config: AttrDict):
        super(NNclrInfoNCELoss, self).__init__()

        self.loss_config = loss_config
        # loss constants
        self.prediction_head_params = self.loss_config.prediction_head_params
        self.temperature = self.loss_config.temperature
        self.queue_size = self.loss_config.queue_size
        self.buffer_params = self.loss_config.buffer_params
        self.criterion = NNclrInfoNCECriterion(
            self.prediction_head_params,
            self.buffer_params,
            self.temperature,
            self.queue_size,
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates NNclrInfoNCELoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            SimclrInfoNCELoss instance.
        """
        return cls(loss_config)

    def forward(self, output, target):
        normalized_output = nn.functional.normalize(output, dim=1, p=2)
        loss = self.criterion(normalized_output)
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name(), "info_average": self.criterion}
        return pprint.pformat(repr_dict, indent=2)


class NNclrInfoNCECriterion(SimclrInfoNCECriterion):
    """
    The criterion corresponding to the NNCLR loss as defined in the https://arxiv.org/abs/2104.14548 paper.

    Args:
        prediction_head_params (dict): prediction MLP parameters
        temperature (float): the temperature to be applied on the logits
        queue_size (int): size of the nearest neighbor support set
        buffer_params:
            world_size (int): total number of trainers in training
            embedding_dim (int): output dimensions of the features projects
            effective_batch_size (int): total batch size used (includes positives)
    """

    def __init__(
        self,
        prediction_head_params: Dict,
        buffer_params: Dict,
        temperature: float,
        queue_size: int,
    ):
        super(NNclrInfoNCECriterion, self).__init__(buffer_params, temperature)

        self.prediction_head_params = prediction_head_params
        self.queue_size = queue_size

        # Create the queue
        self.register_buffer(
            "queue", torch.randn(queue_size, self.buffer_params.embedding_dim)
        )
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.initialized = False

        # Will be set by NNCLRHook
        self.prediction_head = None
        self.preds = None

        # Used if loading a state dict before NNCLRHook had time to initialize the prediction head
        self.checkpoint = None

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            next_ptr = (ptr + batch_size) % self.queue_size

            # replace the keys at ptr (dequeue and enqueue)
            self.queue[ptr:] = keys[next_ptr::]
            self.queue[:next_ptr] = keys[:next_ptr]

            ptr = next_ptr  # move pointer
        else:
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[ptr : ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, embedding: torch.Tensor):
        """
        Calculate the loss. Operates on embeddings tensor.
        """
        assert embedding.ndim == 2
        assert embedding.shape[1] == int(self.buffer_params.embedding_dim)

        batch_size = embedding.shape[0]
        assert (
            batch_size % self.num_pos == 0
        ), "Batch size should be divisible by num_pos"

        if not self.initialized:
            self.queue = self.queue.to(embedding.device)
            self.initialized = True

        with torch.no_grad():
            nearest_neighbors = self.queue[
                torch.cdist(embedding, self.queue, 2).argmin(1)
            ]

        embeddings_buffer = self.gather_embeddings(embedding)

        loss = self._info_nce(nearest_neighbors, embeddings_buffer)

        self._dequeue_and_enqueue(embedding)

        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "prediction_head_params": self.prediction_head_params,
            "temperature": self.temperature,
            "queue_size": self.queue_size,
            "num_negatives": self.buffer_params.effective_batch_size - 2,
            "num_pos": self.num_pos,
            "dist_rank": self.dist_rank,
        }
        return pprint.pformat(repr_dict, indent=2)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Restore the loss state given a checkpoint

        Args:
            state_dict (serialized via torch.save)
        """

        # If the prediction head has been allocated, use the normal pytorch restoration
        if self.prediction_head is None:
            self.checkpoint = state_dict
            logging.info("Storing the checkpoint for later use")
        else:
            logging.info("Restoring checkpoint")
            super().load_state_dict(state_dict, *args, **kwargs)
