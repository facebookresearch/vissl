# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import pprint
from typing import List, Union

import numpy as np
import torch
from classy_vision.generic.distributed_util import (
    all_reduce_mean,
    gather_from_all,
    get_rank,
)
from classy_vision.generic.util import is_pos_int
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict


@register_loss("nce_loss_with_memory")
class NCELossWithMemory(ClassyLoss):
    """
    Distributed version of the NCE loss. It performs an "all_gather" to gather
    the allocated buffers like memory no a single gpu. For this, Pytorch distributed
    backend is used. If using NCCL, one must ensure that all the buffer are on GPU.
    This class supports training using both NCE and CrossEntropy (InfoNCE).

    This loss is used by NPID (https://arxiv.org/pdf/1805.01978.pdf), NPID++ and
    PIRL (https://arxiv.org/abs/1912.01991) approaches.

    Written by: Ishan Misra (imisra@fb.com)

    Config params:
        norm_embedding (bool):           whether to normalize embeddings
        temperature (float):             the temperature to apply to logits
        norm_constant (int):             Z parameter in the NCEAverage
        update_mem_with_emb_index (int): In case we have multiple embeddings used
                                         in the nce loss, specify which embedding
                                         to use to update the memory.
        loss_type (str):                 options are "nce" | "cross_entropy". Using the
                                         cross_entropy turns the loss into InfoNCE loss.
        loss_weights (List[float]):      if the NCE loss is computed between multiple pairs,
                                         we can set a loss weight per term can be used to weight
                                         different pair contributions differently
        negative_sampling_params:
            num_negatives (int):         how many negatives to contrast with
            type (str):                  how to select the negatives. options "random"
        memory_params:
            memory_size (int):           number of training samples as all the samples are
                                         stored in memory
            embedding_dim (int):         the projection head output dimension
            momentum (int):              momentum to use to update the memory
            norm_init (bool):            whether to L2 normalize the initialized memory bank
            update_mem_on_forward (bool): whether to update memory on the forward pass
        num_train_samples (int):         number of unique samples in the training dataset
    """

    def __init__(self, loss_config: AttrDict):
        super(NCELossWithMemory, self).__init__()

        self.loss_config = loss_config
        memory_params = self.loss_config.memory_params
        memory_params.memory_size = self.loss_config.num_train_samples
        assert is_pos_int(
            memory_params.memory_size
        ), f"Memory size must be positive: {memory_params.memory_size}"

        assert self.loss_config.loss_type in [
            "nce",
            "cross_entropy",
        ], f"Supported types are nce/cross_entropy. Found {self.loss_config.loss_type}"

        self.loss_type = self.loss_config.loss_type

        self.update_memory_on_forward = memory_params.update_mem_on_forward
        self.update_memory_emb_index = self.loss_config.update_mem_with_emb_index
        if self.update_memory_on_forward is False:
            # we have multiple embeddings used in NCE
            # but we update memory with only one of them
            assert self.update_memory_emb_index >= 0

        # first setup the NCEAverage method to get the scores of the output wrt
        # memory bank negatives
        self.nce_average = NCEAverage(
            memory_params=memory_params,
            negative_sampling_params=self.loss_config.negative_sampling_params,
            T=self.loss_config.temperature,
            Z=self.loss_config.norm_constant,
            loss_type=self.loss_type,
        )

        if self.loss_type == "nce":
            # setup the actual NCE loss
            self.nce_criterion = NCECriterion(self.loss_config.num_train_samples)
        elif self.loss_type == "cross_entropy":
            # cross-entropy loss. Also called InfoNCE
            self.xe_criterion = nn.CrossEntropyLoss()

        # other constants
        self.normalize_embedding = self.loss_config.norm_embedding
        self.loss_weights = self.loss_config.loss_weights
        self.init_sync_memory = False
        self.ignore_index = self.loss_config.get("ignore_index", -1)

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates NCELossWithMemory from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            NCELossWithMemory instance.
        """
        return cls(loss_config)

    def forward(
        self, output: Union[torch.Tensor, List[torch.Tensor]], target: torch.Tensor
    ):
        """
        For each output and single target, loss is calculated.
        """
        if isinstance(output, torch.Tensor):
            output = [output]
        assert isinstance(
            output, list
        ), "Model output should be a list of tensors. Got Type {}".format(type(output))

        if not self.init_sync_memory:
            self.sync_memory()

        target = target.squeeze()
        # filter out ignore_index ones
        non_ignore = target != self.ignore_index
        target = target[non_ignore]
        output = [x[non_ignore] for x in output]

        loss = 0
        for l_idx, l_output in enumerate(output):
            normalized_output = l_output
            if self.normalize_embedding:
                normalized_output = nn.functional.normalize(l_output, dim=1, p=2)

            if self.update_memory_on_forward is False:
                update_memory_on_forward = self.update_memory_emb_index == l_idx
            else:
                update_memory_on_forward = True
            # compare output embeddings to memory bank embeddings and get scores
            # nce_average is batch x (1 + num_negatives)
            nce_average = self.nce_average(
                normalized_output,
                target,
                update_memory_on_forward=update_memory_on_forward,
            )

            if self.loss_type == "nce":
                curr_loss = self.nce_criterion(nce_average, target)
            elif self.loss_type == "cross_entropy":
                labels = torch.zeros(
                    (nce_average.shape[0], 1),
                    device=nce_average.device,
                    dtype=torch.int64,
                )
                curr_loss = self.xe_criterion(nce_average, labels)
            loss += self.loss_weights[l_idx] * curr_loss
        return loss

    def sync_memory(self):
        """
        Sync memory across all processes before first forward pass. Only needed
        in the distributed case.
        After the first forward pass, the update_memory function in NCEAverage
        does a gather over all embeddings, so memory stays in sync. Doing a gather
        over embeddings is O(batch size). Syncing memory is O(num items in memory).
        Generally, batch size << num items in memory. So, we prefer doing the syncs
        in update_memory.
        """
        self.nce_average.memory = all_reduce_mean(self.nce_average.memory)
        logging.info(f"Rank: {get_rank()}: Memory synced")
        # set to true once we are done. forward pass in nce_average will sync after.
        self.init_sync_memory = True

    def update_memory(self, embedding, y):
        assert (
            self.nce_average.update_memory_on_forward is False
        ), "Memory was already updated on forward"
        with torch.no_grad():
            if self.normalize_embedding:
                embedding = nn.functional.normalize(embedding, dim=1, p=2)
            self.nce_average.update_memory(embedding, y)

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "nce_average": self.nce_average,
            "loss_type": self.loss_type,
            "update_emb_index": self.update_memory_emb_index,
        }
        return pprint.pformat(repr_dict, indent=2)


# Original implementation: https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCEAverage.py   # NOQA
# Adapted by: Ishan Misra (imisra@fb.com)
class NCEAverage(nn.Module):
    """
    Computes the scores of the model embeddings against the `positive'
    and `negative' samples from the Memory Bank.
    This class does *NOT* compute the actual loss, just the scores,
    i.e., inner products followed by normalizations/exponentiation etc.
    """

    def __init__(
        self, memory_params, negative_sampling_params, T=0.07, Z=-1, loss_type="nce"
    ):
        super(NCEAverage, self).__init__()
        self.nLem = memory_params.memory_size
        self.loss_type = loss_type
        self.setup_negative_sampling(negative_sampling_params)
        self.init_memory(memory_params)
        self.register_buffer(
            "params",
            torch.tensor(
                [
                    self.negative_sampling_params.num_negatives,
                    T,
                    Z,
                    self.memory_params.momentum,
                ]
            ),
        )

    def forward(self, embedding, y, idx=None, update_memory_on_forward=None):
        assert embedding.ndim == 2
        assert embedding.shape[1] == self.memory.shape[1]

        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()

        batchSize = embedding.shape[0]
        embedding_dim = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.do_negative_sampling(embedding, y, num_negatives=K)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1)).detach()
        weight = weight.view(batchSize, K + 1, embedding_dim)

        out = torch.bmm(weight, embedding.view(batchSize, embedding_dim, 1))
        out = torch.div(out, T)
        if self.loss_type == "nce":
            out = torch.exp(out)

            # compute partition function
            if Z < 0:
                self.compute_partition_function(out)

            Z = self.params[2].clone().detach().item()

            out = torch.div(out, Z).contiguous().reshape(batchSize, K + 1)
        if update_memory_on_forward or (
            update_memory_on_forward is None and self.update_memory_on_forward
        ):
            self.update_memory(embedding, y)
        return out

    def compute_partition_function(self, out):
        num_items = self.memory.size(0)
        with torch.no_grad():
            batch_mean = out.mean()
            # NOTE: this relies of "mean" computation being stable and deterministic
            # across all nodes. Could be replaced with smarter ways.
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                batch_mean_gathered = gather_from_all(batch_mean)
                all_batch_mean = batch_mean_gathered.mean().squeeze().item()
            else:
                all_batch_mean = batch_mean.item()
        self.params[2] = all_batch_mean * num_items
        Z = self.params[2].clone().detach().item()
        rank = get_rank()
        logging.info(f"Rank: {rank}; Normalization constant Z is set to {Z}")

    def do_negative_sampling(self, embedding, y, num_negatives):
        with torch.no_grad():
            if self.negative_sampling_params.type in ["random", "debug"]:
                batchSize = embedding.shape[0]
                idx = self.multinomial.draw(batchSize * (num_negatives + 1)).view(
                    batchSize, -1
                )
                idx.select(1, 0).copy_(y.data)
        return idx

    def setup_negative_sampling(self, negative_sampling_params):
        self.negative_sampling_params = negative_sampling_params
        assert self.negative_sampling_params["type"] in ["random", "debug"]
        self.negative_sampling_params = negative_sampling_params
        if self.negative_sampling_params.type == "debug":
            logging.info("Using debug mode for negative sampling.")
            logging.info("Will use slower NumpySampler.")
            self.multinomial = NumpySampler(self.nLem)
        else:
            unigrams = torch.ones(self.nLem)
            self.multinomial = AliasMethod(unigrams)
        self.num_negatives = negative_sampling_params["num_negatives"]

    def init_memory(self, memory_params):
        self.memory_params = memory_params
        num_items = memory_params.memory_size
        embedding_dim = memory_params.embedding_dim
        self.update_memory_on_forward = memory_params.update_mem_on_forward
        stdv = 1.0 / math.sqrt(embedding_dim / 3)
        self.register_buffer(
            "memory", torch.rand(num_items, embedding_dim).mul_(2 * stdv).add_(-stdv)
        )

        if memory_params.norm_init:
            self.memory = nn.functional.normalize(self.memory, p=2, dim=1)
        sample_norm = self.memory[:10].norm(dim=1).mean()
        mem_info = f"Init memory: {self.memory.shape}; \
            stdv: {stdv}; normalize: {memory_params.norm_init} \
            norm: {sample_norm}"
        logging.info(f"Rank: {get_rank()} - {mem_info}")

    def update_memory(self, embedding, y):
        momentum = self.params[3].item()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            embedding_gathered = gather_from_all(embedding)
            y_gathered = gather_from_all(y)
        else:
            embedding_gathered = embedding
            y_gathered = y

        # update memory
        with torch.no_grad():
            # Assumption: memory_size >= y.max()
            assert y_gathered.max() < self.memory.shape[0], (
                f"Memory bank {self.memory.shape} is not sufficient "
                f"to hold index: {y_gathered.max()}"
            )
            l_pos = torch.index_select(self.memory, 0, y_gathered.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(embedding_gathered, 1 - momentum))
            updated_l = nn.functional.normalize(l_pos, p=2, dim=1)
            self.memory.index_copy_(0, y_gathered, updated_l)

    def __repr__(self):
        num_negatives = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()
        repr_dict = {
            "name": self._get_name(),
            "T": T,
            "Z": Z,
            "num_negatives": num_negatives,
            "momentum": momentum,
            "memory_buffer_size": self.memory.shape,
            "negative_sampling": self.negative_sampling_params,
            "memory": self.memory_params,
            "update_memory_on_forward": self.update_memory_on_forward,
        }
        return pprint.pformat(repr_dict, indent=2)


# Credits: https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/alias_multinomial.property   # NOQA
class AliasMethod(nn.Module):
    """
    A fast way to sample from a multinomial distribution.
    Faster than torch.multinomial or np.multinomial.
    The setup (__init__) for this class is slow, however
    `draw' (actual sampling) is fast.
    """

    def __init__(self, probs):
        super(AliasMethod, self).__init__()
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.register_buffer("prob", torch.zeros(K))
        self.register_buffer("alias", torch.LongTensor([0] * K))

        # Sort the data into the outcomes with probabilities that are larger and
        # smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that appropriately allocate
        # the larger outcomes over the overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def draw(self, N):
        """
        Draw N samples from multinomial
        :param N: number of samples
        :return: samples
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj


# Numpy based sampler. Useful for debugging
# This Sampler is faster to setup than AliasMethod.
# However, it is **much** slower to sample from it.
class NumpySampler:
    def __init__(self, high):
        self.high = high

    def draw(self, num_negatives):
        rand_nums = np.random.choice(self.high, size=num_negatives)
        rand_nums = torch.from_numpy(rand_nums).long().cuda()
        return rand_nums


# Core NCE Criterion that computes cross entropy with a fixed prior on negatives
# from lemniscate
# Credits: https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCECriterion.property   # NOQA
class NCECriterion(nn.Module):
    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem
        self.eps = float(np.finfo("float32").eps)

    def forward(self, x, targets):
        batchSize = x.size(0)
        K = x.size(1) - 1
        Pnt = 1 / float(self.nLem)
        Pns = 1 / float(self.nLem)

        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
        Pmt = x.select(1, 0)
        Pmt_div = Pmt.add(K * Pnt + self.eps)
        lnPmt = torch.div(Pmt, Pmt_div)

        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon_div = x.narrow(1, 1, K).add(K * Pns + self.eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)

        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)

        loss = -(lnPmtsum + lnPonsum) / batchSize

        return loss
