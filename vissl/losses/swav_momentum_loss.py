# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import pprint

import numpy as np
import torch
from classy_vision.generic.distributed_util import (
    all_reduce_sum,
    get_cuda_device_index,
    get_world_size,
    is_distributed_training_run,
)
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict


@register_loss("swav_momentum_loss")
class SwAVMomentumLoss(ClassyLoss):
    """
    This loss extends the SwAV loss proposed in paper https://arxiv.org/abs/2006.09882
    by Caron et al. The loss combines the benefits of using the SwAV approach
    with the momentum encoder as used in MoCo.

    Config params:
        momentum (float):               for the momentum encoder
        momentum_eval_mode_iter_start (int): from what iteration should the momentum encoder
                                        network be in eval mode
        embedding_dim (int):            the projection head output dimension
        temperature (float):            temperature to be applied to the logits
        use_double_precision (bool):    whether to use double precision for the loss.
                                        This could be a good idea to avoid NaNs.
        normalize_last_layer (bool):    whether to normalize the last layer
        num_iters (int):                number of sinkhorn algorithm iterations to make
        epsilon (float):                see the paper for details
        num_crops (int):                number of crops used
        crops_for_assign (List[int]):   what crops to use for assignment
        num_prototypes (List[int]):     number of prototypes
        queue:
            queue_length (int):         number of features to store and used in the scores
            start_iter (int):           when to start using the queue for the scores
            local_queue_length (int):   length of queue per gpu
    """

    def __init__(self, loss_config: AttrDict):
        super().__init__()
        self.loss_config = loss_config

        self.momentum_encoder = None
        self.checkpoint = None
        self.momentum_scores = None
        self.momentum_embeddings = None
        self.is_distributed = is_distributed_training_run()
        self.use_gpu = get_cuda_device_index() > -1
        self.softmax = nn.Softmax(dim=1)

        # keep track of number of iterations
        self.register_buffer("num_iteration", torch.zeros(1, dtype=int))

        # for queue
        self.use_queue = False
        if self.loss_config.queue.local_queue_length > 0:
            self.initialize_queue()

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates SwAVMomentumLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            SwAVMomentumLoss instance.
        """
        return cls(loss_config)

    def initialize_queue(self):
        for i, nmb_proto in enumerate(self.loss_config.num_prototypes):
            init_queue = (
                torch.rand(
                    len(self.loss_config.crops_for_assign),
                    self.loss_config.queue.local_queue_length,
                    nmb_proto,
                )
                * 2
                - 1
            )
            self.register_buffer("local_queue" + str(i), init_queue)
        stdv = 1.0 / math.sqrt(self.loss_config.embedding_dim / 3)
        init_queue = (
            torch.rand(
                len(self.loss_config.crops_for_assign),
                self.loss_config.queue.local_queue_length,
                self.loss_config.embedding_dim,
            )
            .mul_(2 * stdv)
            .add_(-stdv)
        )
        self.register_buffer("local_emb_queue", init_queue)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Restore the loss state given a checkpoint

        Args:
            state_dict (serialized via torch.save)
        """

        # If the encoder has been allocated, use the normal pytorch restoration
        if self.momentum_encoder is None:
            self.checkpoint = state_dict
            logging.info("Storing the checkpoint for later use")
        else:
            logging.info("Restoring checkpoint")
            super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, output: torch.Tensor, *args, **kwargs):
        self.use_queue = (
            self.loss_config.queue.local_queue_length > 0
            and self.num_iteration >= self.loss_config.queue.start_iter
        )
        if self.use_queue:
            if self.is_distributed:
                self.compute_queue_scores(self.momentum_encoder.module.heads[0])
            else:
                self.compute_queue_scores(self.momentum_encoder.heads[0])

        loss = 0
        for head_id, proto_scores in enumerate(output[1:]):

            bs = proto_scores.shape[0] // self.loss_config.num_crops
            sub_loss = 0
            for j, crop_id in enumerate(self.loss_config.crops_for_assign):
                with torch.no_grad():
                    scores_this_crop = self.momentum_scores[head_id][
                        j * bs : (j + 1) * bs
                    ]
                    if self.use_queue:
                        queue = getattr(self, "local_queue" + str(head_id))[j].clone()
                        scores_this_crop = torch.cat((scores_this_crop, queue))
                    assignments = torch.exp(
                        scores_this_crop / self.loss_config.epsilon
                    ).t()
                    assignments = self.distributed_sinkhornknopp(assignments)[:bs]
                idx_crop_pred = np.delete(
                    np.arange(self.loss_config.num_crops), crop_id
                )
                subsubloss = 0
                for p in idx_crop_pred:
                    subsubloss -= torch.mean(
                        torch.sum(
                            assignments
                            * torch.log(
                                self.softmax(
                                    proto_scores[bs * p : bs * (p + 1)]
                                    / self.loss_config.temperature
                                )
                            ),
                            dim=1,
                        )
                    )
                sub_loss += subsubloss / len(idx_crop_pred)
            loss += sub_loss / len(self.loss_config.crops_for_assign)
        loss /= len(output) - 1

        self.num_iteration += 1
        if self.use_queue:
            self.update_emb_queue()

        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)

    def distributed_sinkhornknopp(self, Q: torch.Tensor):
        """
        Apply the distributed sinknorn optimization on the scores matrix to
        find the assignments
        """
        with torch.no_grad():
            sum_Q = torch.sum(Q, dtype=Q.dtype)
            all_reduce_sum(sum_Q)
            Q /= sum_Q

            k = Q.shape[0]
            n = Q.shape[1]
            N = get_world_size() * Q.shape[1]

            # we follow the u, r, c and Q notations from
            # https://arxiv.org/abs/1911.05371
            r = torch.ones(k) / k
            c = torch.ones(n) / N

            if self.use_gpu:
                r = r.cuda(non_blocking=True)
                c = c.cuda(non_blocking=True)

            curr_sum = torch.sum(Q, dim=1, dtype=Q.dtype)
            all_reduce_sum(curr_sum)

            for _ in range(self.loss_config.num_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0, dtype=Q.dtype)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1, dtype=Q.dtype)
                all_reduce_sum(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True, dtype=Q.dtype)).t().float()

    def update_emb_queue(self):
        with torch.no_grad():
            bs = len(self.momentum_embeddings) // self.loss_config.num_crops
            for i in range(len(self.loss_config.crops_for_assign)):
                queue = self.local_emb_queue[i]
                queue[bs:] = queue[:-bs].clone()
                queue[:bs] = self.momentum_embeddings[i * bs : (i + 1) * bs]
                self.local_emb_queue[i] = queue

    def compute_queue_scores(self, head):
        with torch.no_grad():
            for i in range(len(self.loss_config.crops_for_assign)):
                for h in range(head.nmb_heads):
                    scores = getattr(head, "prototypes" + str(h))(
                        self.local_emb_queue[i]
                    )
                    getattr(self, "local_queue" + str(h))[i] = scores
