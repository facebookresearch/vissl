# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import math
import os
import pprint
from typing import List

import numpy as np
import torch
from classy_vision.generic.distributed_util import (
    all_reduce_max,
    all_reduce_sum,
    get_cuda_device_index,
    get_rank,
    get_world_size,
)
from classy_vision.losses import ClassyLoss, register_loss
from fvcore.common.file_io import PathManager
from torch import nn
from vissl.utils.hydra_config import AttrDict


@register_loss("swav_loss")
class SwAVLoss(ClassyLoss):
    """
    This loss is proposed by the SwAV paper https://arxiv.org/abs/2006.09882
    by Caron et al. See the paper for more details about the loss.

    Config params:
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
        temp_hard_assignment_iters (int): whether to do hard assignment for the initial
                                        few iterations
        output_dir (str):               for dumping the debugging info in case loss
                                        becomes NaN
        queue:
            queue_length (int):         number of features to store and used in the scores
            start_iter (int):           when to start using the queue for the scores
            local_queue_length (int):   length of queue per gpu
    """

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
            self.loss_config.temp_hard_assignment_iters,
            self.loss_config.output_dir,
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates SwAVLoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            SwAVLoss instance.
        """
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
        repr_dict = {
            "name": self._get_name(),
            "epsilon": self.loss_config.epsilon,
            "use_double_precision": self.loss_config.use_double_precision,
            "local_queue_length": self.loss_config.queue.local_queue_length,
            "temperature": self.loss_config.temperature,
            "num_prototypes": self.loss_config.num_prototypes,
            "num_crops": self.loss_config.num_crops,
            "nmb_sinkhornknopp_iters": self.loss_config.num_iters,
            "embedding_dim": self.loss_config.embedding_dim,
            "temp_hard_assignment_iters": self.loss_config.temp_hard_assignment_iters,
        }
        return pprint.pformat(repr_dict, indent=2)


class SwAVCriterion(nn.Module):
    """
    This criterion is used by the SwAV paper https://arxiv.org/abs/2006.09882
    by Caron et al. See the paper for more details about the loss.

    Config params:
        embedding_dim (int):            the projection head output dimension
        temperature (float):            temperature to be applied to the logits

        num_iters (int):                number of sinkhorn algorithm iterations to make
        epsilon (float):                see the paper for details
        num_crops (int):                number of crops used
        crops_for_assign (List[int]):   what crops to use for assignment
        num_prototypes (List[int]):     number of prototypes
        temp_hard_assignment_iters (int): whether to do hard assignment for the initial
                                        few iterations
        output_dir (str):               for dumping the debugging info in case loss
                                        becomes NaN
        local_queue_length (int):   length of queue per gpu
    """

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
        temp_hard_assignment_iters: int,
        output_dir: str,
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
        self.temp_hard_assignment_iters = temp_hard_assignment_iters
        self.local_queue_length = local_queue_length
        self.dist_rank = get_rank()
        self.world_size = get_world_size()
        self.log_softmax = nn.LogSoftmax(dim=1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.register_buffer("num_iteration", torch.zeros(1, dtype=int))
        self.use_queue = False
        if local_queue_length > 0:
            self.initialize_queue()
        self.output_dir = output_dir

    def distributed_sinkhornknopp(self, Q: torch.Tensor):
        """
        Apply the distributed sinknorn optimization on the scores matrix to
        find the assignments
        """
        eps_num_stab = 1e-12
        with torch.no_grad():
            # remove potential infs in Q
            # replace the inf entries with the max of the finite entries in Q
            mask = torch.isinf(Q)
            ind = torch.nonzero(mask)
            if len(ind) > 0:
                for i in ind:
                    Q[i[0], i[1]] = 0
                m = torch.max(Q)
                for i in ind:
                    Q[i[0], i[1]] = m
            sum_Q = torch.sum(Q, dtype=Q.dtype)
            all_reduce_sum(sum_Q)
            Q /= sum_Q

            k = Q.shape[0]
            n = Q.shape[1]
            N = self.world_size * Q.shape[1]

            # we follow the u, r, c and Q notations from
            # https://arxiv.org/abs/1911.05371
            r = torch.ones(k) / k
            c = torch.ones(n) / N
            if self.use_double_prec:
                r, c = r.double(), c.double()

            if self.use_gpu:
                r = r.cuda(non_blocking=True)
                c = c.cuda(non_blocking=True)

            for _ in range(self.nmb_sinkhornknopp_iters):
                u = torch.sum(Q, dim=1, dtype=Q.dtype)
                all_reduce_sum(u)

                # for numerical stability, add a small epsilon value
                # for non-zero Q values.
                if len(torch.nonzero(u == 0)) > 0:
                    Q += eps_num_stab
                    u = torch.sum(Q, dim=1, dtype=Q.dtype)
                    all_reduce_sum(u)
                u = r / u
                # remove potential infs in "u"
                # replace the inf entries with the max of the finite entries in "u"
                mask = torch.isinf(u)
                ind = torch.nonzero(mask)
                if len(ind) > 0:
                    for i in ind:
                        u[i[0]] = 0
                    m = torch.max(u)
                    for i in ind:
                        u[i[0]] = m

                Q *= u.unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0, dtype=Q.dtype)).unsqueeze(0)
            Q = (Q / torch.sum(Q, dim=0, keepdim=True, dtype=Q.dtype)).t().float()

            # hard assignment
            if self.num_iteration < self.temp_hard_assignment_iters:
                index_max = torch.max(Q, dim=1)[1]
                Q.zero_()
                Q.scatter_(1, index_max.unsqueeze(1), 1)
            return Q

    def forward(self, scores: torch.Tensor, head_id: int):
        assert scores.shape[0] % self.num_crops == 0
        bs = scores.shape[0] // self.num_crops

        total_loss = 0
        n_term_loss = 0

        # 2 big crops are normally used for the assignment
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
                    assignments = scores_this_crop / self.epsilon
                    # use the log-sum-exp trick for numerical stability.
                    M = torch.max(assignments)
                    all_reduce_max(M)
                    assignments -= M
                    assignments = torch.exp(assignments).t()
                assignments = self.distributed_sinkhornknopp(assignments)[:bs]
                idx_crop_pred = np.delete(np.arange(self.num_crops), crop_id)

            loss = 0
            for p in idx_crop_pred:
                if self.use_double_prec:
                    loss -= torch.mean(
                        torch.sum(
                            assignments
                            * self.log_softmax(
                                scores[bs * p : bs * (p + 1)].double()
                                / np.float64(self.temperature)
                            ),
                            dim=1,
                            dtype=assignments.dtype,
                        )
                    )
                else:
                    loss -= torch.mean(
                        torch.sum(
                            assignments
                            * self.log_softmax(
                                scores[bs * p : bs * (p + 1)] / self.temperature
                            ),
                            dim=1,
                            dtype=assignments.dtype,
                        )
                    )
            loss /= len(idx_crop_pred)
            total_loss += loss
            n_term_loss += 1

            # stop training if NaN appears and log the output to help debugging
            # TODO (prigoyal): extract the logic to be common for all losses
            # debug_state() method that all losses can override
            if torch.isnan(loss):
                logging.info(
                    f"Infinite Loss or NaN. Loss value: {loss}, rank: {self.dist_rank}"
                )
                scores_output_file = os.path.join(
                    self.output_dir,
                    "rank" + str(self.dist_rank) + "_scores" + str(i) + ".pth",
                )
                assignments_out_file = os.path.join(
                    self.output_dir,
                    "rank" + str(self.dist_rank) + "_assignments" + str(i) + ".pth",
                )
                with PathManager.open(scores_output_file, "wb") as fwrite:
                    torch.save(scores, fwrite)
                with PathManager.open(assignments_out_file, "wb") as fwrite:
                    torch.save(assignments, fwrite)
                logging.info(f"Saved the scores matrix to: {scores_output_file}")
                logging.info(f"Saved the assignment matrix to: {assignments_out_file}")
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
        repr_dict = {
            "name": self._get_name(),
            "use_queue": self.use_queue,
            "local_queue_length": self.local_queue_length,
            "temperature": self.temperature,
            "num_prototypes": self.num_prototypes,
            "num_crops": self.num_crops,
            "nmb_sinkhornknopp_iters": self.nmb_sinkhornknopp_iters,
            "embedding_dim": self.embedding_dim,
            "temp_hard_assignment_iters": self.temp_hard_assignment_iters,
        }
        return pprint.pformat(repr_dict, indent=2)
