# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math
import pprint
from typing import List

import numpy as np
import torch
from classy_vision.generic.distributed_util import (
    all_reduce_sum,
    get_cuda_device_index,
    get_rank,
    get_world_size,
)
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.utils.hydra_config import AttrDict


@register_loss("swav_loss")
class SwAVLoss(ClassyLoss):
    def __init__(self, loss_config: AttrDict):
        super().__init__()

        self.loss_config = loss_config
        self.queue_start_iter = self.loss_config.queue.start_iter
        self.swav_criterion = SwAVCriterion(
            self.loss_config.temperature,
            self.loss_config.crops_for_assign,
            self.loss_config.num_crops,
            self.loss_config.num_iters,
            self.loss_config.epsilon,
            self.loss_config.use_double_precision,
            self.loss_config.num_prototypes,
            self.loss_config.queue.local_queue_length,
            self.loss_config.embedding_dim,
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        return cls(loss_config)

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        self.swav_criterion.use_queue = (
            self.swav_criterion.local_queue_length > 0
            and self.swav_criterion.num_iteration >= self.queue_start_iter
        )
        loss = 0
        for i, prototypes_scores in enumerate(output[1:]):
            loss += self.swav_criterion(prototypes_scores, i)
        loss /= len(output) - 1
        self.swav_criterion.num_iteration += 1
        if self.swav_criterion.use_queue:
            self.swav_criterion.update_emb_queue(output[0].detach())
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)


class SwAVCriterion(nn.Module):
    def __init__(
        self,
        temperature: float,
        crops_for_assign: List[int],
        num_crops: int,
        num_iters: int,
        epsilon: float,
        use_double_prec: bool,
        num_prototypes: List[int],
        local_queue_length: int,
        embedding_dim: int,
    ):
        super(SwAVCriterion, self).__init__()

        self.use_gpu = get_cuda_device_index() > -1

        self.temperature = temperature
        self.crops_for_assign = crops_for_assign
        self.num_crops = num_crops
        self.nmb_sinkhornknopp_iters = num_iters
        self.epsilon = epsilon
        self.use_double_prec = use_double_prec
        self.num_prototypes = num_prototypes
        self.nmb_heads = len(self.num_prototypes)
        self.embedding_dim = embedding_dim
        self.local_queue_length = local_queue_length
        self.dist_rank = get_rank()
        self.world_size = get_world_size()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.register_buffer("num_iteration", torch.zeros(1, dtype=int))
        self.use_queue = False
        if local_queue_length > 0:
            self.initialize_queue()

    def distributed_sinkhornknopp(self, Q: torch.Tensor):
        with torch.no_grad():
            sum_Q = torch.sum(Q, dtype=Q.dtype)
            all_reduce_sum(sum_Q)
            Q /= sum_Q

            k = Q.shape[0]
            n = Q.shape[1]
            N = self.world_size * Q.shape[1]

            # we follow the u, r, c and Q notations from
            # https://arxiv.org/abs/1911.05371
            u = torch.zeros(k)
            r = torch.ones(k) / k
            c = torch.ones(n) / N
            if self.use_double_prec:
                u, r, c = u.double(), r.double(), c.double()

            if self.use_gpu:
                u = u.cuda(non_blocking=True)
                r = r.cuda(non_blocking=True)
                c = c.cuda(non_blocking=True)

            curr_sum = torch.sum(Q, dim=1, dtype=Q.dtype)
            all_reduce_sum(curr_sum)

            for _ in range(self.nmb_sinkhornknopp_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0, dtype=Q.dtype)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1, dtype=Q.dtype)
                all_reduce_sum(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True, dtype=Q.dtype)).t().float()

    def forward(self, scores: torch.Tensor, head_id: int):
        assert scores.shape[0] % self.num_crops == 0
        bs = scores.shape[0] // self.num_crops

        total_loss = 0
        n_term_loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                scores_this_crop = scores[bs * crop_id : bs * (crop_id + 1)]
                if self.use_queue:
                    queue = getattr(self, "local_queue" + str(head_id))[i].clone()
                    scores_this_crop = torch.cat((scores_this_crop, queue))
                if self.use_double_prec:
                    assignments = torch.exp(
                        scores_this_crop.double() / np.float64(self.epsilon)
                    ).t()
                    assignments = assignments.double()
                else:
                    assignments = torch.exp(scores_this_crop / self.epsilon).t()
                assignments = self.distributed_sinkhornknopp(assignments)[:bs]
                idx_crop_pred = np.delete(np.arange(self.num_crops), crop_id)
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
            bs = len(emb) // self.num_crops
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
                    self.num_prototypes[i],
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
