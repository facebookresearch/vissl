# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import pprint

import numpy as np
import torch
from classy_vision.generic.distributed_util import (
    gather_from_all,
    get_cuda_device_index,
    get_rank,
)
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn


@register_loss("simclr_info_nce_loss")
class DistributedSimclrInfoNCELoss(ClassyLoss):
    def __init__(self, config, device: str = "gpu"):
        super(DistributedSimclrInfoNCELoss, self).__init__()

        self.loss_config = config.SIMCLR_INFO_NCE_LOSS
        # loss constants
        self.temperature = self.loss_config.TEMPERATURE
        self.buffer_params = self.loss_config.BUFFER_PARAMS
        self.info_criterion = SimclrInfoNCECriterion(
            self.buffer_params, self.temperature
        )

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def forward(self, output, target):
        assert isinstance(
            output, list
        ), "Model output should be a list of tensors. Got Type {}".format(type(output))

        loss = 0
        for l_output in output:
            normalized_output = nn.functional.normalize(l_output, dim=1, p=2)
            loss += self.info_criterion(normalized_output)
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name(), "info_average": self.info_criterion}
        return pprint.pformat(repr_dict, indent=2)


class SimclrInfoNCECriterion(nn.Module):
    def __init__(self, buffer_params, temperature: float):
        super(SimclrInfoNCECriterion, self).__init__()

        self.use_gpu = get_cuda_device_index() > -1
        self.temperature = temperature
        self.num_pos = 2
        self.buffer_params = buffer_params
        self.criterion = nn.CrossEntropyLoss()
        self.dist_rank = get_rank()
        self.pos_mask = None
        self.neg_mask = None
        self.precompute_pos_neg_mask()
        logging.info(f"Creating Info-NCE loss on Rank: {self.dist_rank}")

    def precompute_pos_neg_mask(self):
        # computed once at the begining of training
        total_images = self.buffer_params.EFFECTIVE_BATCH_SIZE
        world_size = self.buffer_params.WORLD_SIZE
        batch_size = total_images // world_size
        orig_images = batch_size // self.num_pos
        rank = self.dist_rank

        pos_mask = torch.zeros(batch_size, total_images)
        neg_mask = torch.zeros(batch_size, total_images)

        all_indices = np.arange(total_images)
        pos_members = orig_images * np.arange(self.num_pos)
        orig_members = torch.arange(orig_images)
        for anchor in np.arange(self.num_pos):
            for img_idx in range(orig_images):
                delete_inds = batch_size * rank + img_idx + pos_members
                neg_inds = torch.tensor(np.delete(all_indices, delete_inds)).long()
                neg_mask[anchor * orig_images + img_idx, neg_inds] = 1
            for pos in np.delete(np.arange(self.num_pos), anchor):
                pos_inds = batch_size * rank + pos * orig_images + orig_members
                pos_mask[
                    torch.arange(
                        anchor * orig_images, (anchor + 1) * orig_images
                    ).long(),
                    pos_inds.long(),
                ] = 1
        self.pos_mask = pos_mask.cuda(non_blocking=True) if self.use_gpu else pos_mask
        self.neg_mask = neg_mask.cuda(non_blocking=True) if self.use_gpu else neg_mask

    def forward(self, embedding):
        assert embedding.ndim == 2
        assert embedding.shape[1] == int(self.buffer_params.EMBEDDING_DIM)

        batch_size = embedding.shape[0]
        T = self.temperature
        num_pos = self.num_pos
        assert batch_size % num_pos == 0, "Batch size should be divisible by num_pos"

        # Step 1: gather all the embeddings. Shape example: 4096 x 128
        embeddings_buffer = self.gather_embeddings(embedding)

        # Step 2: matrix multiply: 64 x 128 with 4096 x 128 = 64 x 4096 and
        # divide by temperature.
        similarity = torch.exp(torch.mm(embedding, embeddings_buffer.t()) / T)
        pos = torch.sum(similarity * self.pos_mask, 1)
        neg = torch.sum(similarity * self.neg_mask, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss

    def __repr__(self):
        num_negatives = self.buffer_params.EFFECTIVE_BATCH_SIZE - 2
        T = self.temperature
        num_pos = self.num_pos
        repr_dict = {
            "name": self._get_name(),
            "temperature": T,
            "num_negatives": num_negatives,
            "num_pos": num_pos,
            "dist_rank": self.dist_rank,
        }
        return pprint.pformat(repr_dict, indent=2)

    def gather_embeddings(self, embedding):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            embedding_gathered = gather_from_all(embedding)
        else:
            embedding_gathered = embedding
        return embedding_gathered
