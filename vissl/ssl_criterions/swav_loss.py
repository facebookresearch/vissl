#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import pprint

import numpy as np
import torch
from classy_vision.generic.distributed_util import (
    all_reduce_sum,
    get_rank,
    get_world_size,
)
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn


@register_loss("swav_loss")
class SwAVLoss(ClassyLoss):
    def __init__(self, config):
        super().__init__()

        self.loss_config = config.SWAV_LOSS
        self.queue_start_iter = self.loss_config.QUEUE.START_ITER
        self.swav_criterion = SwAVCriterion(
            self.loss_config.TEMPERATURE,
            self.loss_config.CROPS_FOR_ASSIGN,
            self.loss_config.NMB_CROPS,
            self.loss_config.NMB_ITERS,
            self.loss_config.EPSILON,
            self.loss_config.USE_DOUBLE_PRECISION,
            self.loss_config.NMB_PROTOTYPES,
            self.loss_config.QUEUE.LOCAL_QUEUE_LENGTH,
            self.loss_config.EMBEDDING_DIM,
        )

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def forward(self, output, target):
        assert isinstance(
            output, list
        ), "Model output should be a list of tensors. Got Type {}".format(type(output))

        loss = 0
        self.swav_criterion.use_queue = (
            self.swav_criterion.local_queue_length > 0
            and self.swav_criterion.num_iteration >= self.queue_start_iter
        )

        assert len(output) == 1
        for i, prototypes_scores in enumerate(output[0][1:]):
            loss += self.swav_criterion(prototypes_scores, i)
        loss /= len(output[0]) - 1
        self.swav_criterion.num_iteration += 1
        if self.swav_criterion.use_queue:
            self.swav_criterion.update_emb_queue(output[0][0].detach())
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)


class SwAVCriterion(nn.Module):
    def __init__(
        self,
        temperature,
        crops_for_assign,
        nmb_crops,
        nmb_iters,
        epsilon,
        use_double_prec,
        nmb_prototypes,
        local_queue_length,
        embedding_dim,
    ):
        super(SwAVCriterion, self).__init__()

        self.temperature = temperature
        self.crops_for_assign = crops_for_assign
        self.nmb_crops = nmb_crops
        self.nmb_sinkhornknopp_iters = nmb_iters
        self.epsilon = epsilon
        self.use_double_prec = use_double_prec
        self.nmb_prototypes = nmb_prototypes
        self.nmb_heads = len(self.nmb_prototypes)
        self.embedding_dim = embedding_dim
        self.local_queue_length = local_queue_length
        self.dist_rank = get_rank()
        self.world_size = get_world_size()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.register_buffer("num_iteration", torch.zeros(1, dtype=int))
        self.use_queue = False
        if local_queue_length > 0:
            self.initialize_queue()

    def distributed_sinkhornknopp(self, Q):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            all_reduce_sum(sum_Q)
            Q /= sum_Q

            k = Q.shape[0]
            n = Q.shape[1]
            N = self.world_size * Q.shape[1]

            # we follow the u, r, c and Q notations from
            # https://arxiv.org/abs/1911.05371
            u = torch.zeros(k).cuda(non_blocking=True)
            r = torch.ones(k).cuda(non_blocking=True) / k
            c = torch.ones(n).cuda(non_blocking=True) / N
            if self.use_double_prec:
                u, r, c = u.double(), r.double(), c.double()

            curr_sum = torch.sum(Q, dim=1)
            all_reduce_sum(curr_sum)

            for _ in range(self.nmb_sinkhornknopp_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                all_reduce_sum(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def forward(self, scores, head_id):
        assert scores.shape[0] % self.nmb_crops == 0
        bs = scores.shape[0] // self.nmb_crops

        total_loss = 0
        n_term_loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                scores_this_crop = scores[bs * crop_id : bs * (crop_id + 1)]
                if self.use_queue:
                    queue = getattr(self, "local_queue" + str(head_id))[i].clone()
                    scores_this_crop = torch.cat((scores_this_crop, queue))
                assignments = torch.exp(scores_this_crop / self.epsilon).t()
                if self.use_double_prec:
                    assignments = assignments.double()
                assignments = self.distributed_sinkhornknopp(assignments)[:bs]
                idx_crop_pred = np.delete(np.arange(self.nmb_crops), crop_id)
            loss = 0
            for p in idx_crop_pred:
                loss -= torch.mean(
                    torch.sum(
                        assignments
                        * torch.log(
                            self.softmax(
                                scores[bs * p : bs * (p + 1)] / self.temperature
                            )
                        ),
                        dim=1,
                    )
                )
            loss /= len(idx_crop_pred)
            total_loss += loss
            n_term_loss += 1
        total_loss /= n_term_loss
        return total_loss

    def update_emb_queue(self, emb):
        with torch.no_grad():
            bs = len(emb) // self.nmb_crops
            for i, crop_id in enumerate(self.crops_for_assign):
                queue = self.local_emb_queue[i]
                queue[bs:] = queue[:-bs].clone()
                queue[:bs] = emb[crop_id * bs : (crop_id + 1) * bs]
                self.local_emb_queue[i] = queue

    def compute_queue_scores(self, head):
        with torch.no_grad():
            for crop_id in range(len(self.crops_for_assign)):
                for i in range(head.nmb_heads):
                    scores = getattr(head, "prototypes" + str(i))(
                        self.local_emb_queue[crop_id]
                    )
                    getattr(self, "local_queue" + str(i))[crop_id] = scores

    def initialize_queue(self):
        for i in range(self.nmb_heads):
            init_queue = (
                torch.rand(
                    len(self.crops_for_assign),
                    self.local_queue_length,
                    self.nmb_prototypes[i],
                )
                * 2
                - 1
            )
            self.register_buffer("local_queue" + str(i), init_queue)
        stdv = 1.0 / math.sqrt(self.embedding_dim / 3)
        init_queue = (
            torch.rand(
                len(self.crops_for_assign), self.local_queue_length, self.embedding_dim
            )
            .mul_(2 * stdv)
            .add_(-stdv)
        )
        self.register_buffer("local_emb_queue", init_queue)

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)
