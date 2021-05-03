# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint

import numpy as np
import torch
from classy_vision.losses import register_loss
from vissl.config import AttrDict
from vissl.losses.simclr_info_nce_loss import SimclrInfoNCECriterion, SimclrInfoNCELoss


@register_loss("multicrop_simclr_info_nce_loss")
class MultiCropSimclrInfoNCELoss(SimclrInfoNCELoss):
    """
    Expanded version of the SimCLR loss. The SimCLR loss works only on 2 positives.
    We expand the loss to work for more positives following the multi-crop
    augmentation proposed in SwAV paper. See SwAV paper https://arxiv.org/abs/2006.09882
    for the multi-crop augmentation details.

    Config params:
        temperature (float): the temperature to be applied on the logits
        num_crops (int): number of positives used
        buffer_params:
            world_size (int): total number of trainers in training
            embedding_dim (int): output dimensions of the features projects
            effective_batch_size (int): total batch size used (includes positives)
    """

    def __init__(self, loss_config: AttrDict, device: str = "gpu"):
        super(SimclrInfoNCELoss, self).__init__()

        self.loss_config = loss_config
        # loss constants
        self.temperature = self.loss_config.temperature
        self.buffer_params = self.loss_config.buffer_params
        self.num_crops = self.loss_config.num_crops
        self.info_criterion = MultiCropSimclrInfoNCECriterion(
            self.buffer_params, self.temperature, self.num_crops
        )

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "info_average": self.info_criterion,
            "num_crops": self.num_crops,
        }
        return pprint.pformat(repr_dict, indent=2)


class MultiCropSimclrInfoNCECriterion(SimclrInfoNCECriterion):
    """
    The criterion corresponding to the expandion SimCLR loss (as defined in the paper
    https://arxiv.org/abs/2002.05709) using the multi-crop augmentaion proposed
    in SwAV paper. The multi-crop augmentation allows using more positives
    per image.

    Args:
        temperature (float): the temperature to be applied on the logits
        num_crops (int): number of positives
        buffer_params:
            world_size (int): total number of trainers in training
            embedding_dim (int): output dimensions of the features projects
            effective_batch_size (int): total batch size used (includes positives)
    """

    def __init__(self, buffer_params, temperature: float, num_crops: int):

        self.num_crops = num_crops
        super(MultiCropSimclrInfoNCECriterion, self).__init__(
            buffer_params, temperature
        )
        logging.info(f"Setting multicrop num_crops: {num_crops}")

    def precompute_pos_neg_mask(self):
        """
        We precompute the positive and negative masks to speed up the loss calculation
        """
        # computed once at the beginning of training
        total_images = self.buffer_params.effective_batch_size
        world_size = self.buffer_params.world_size
        local_orig_images = total_images // world_size
        local_crops = local_orig_images * self.num_crops
        rank = self.dist_rank

        pos_temps = []
        for d in np.arange(self.num_crops):
            pos_temp, neg_temp = [], []
            for i in range(world_size):
                if i == rank:
                    pos = np.eye(local_crops, k=d * local_orig_images) + np.eye(
                        local_crops, k=-local_crops + d * local_orig_images
                    )
                    neg = np.ones((local_crops, local_crops))
                else:
                    pos = np.zeros((local_crops, local_crops))
                    neg = np.zeros((local_crops, local_crops))
                pos_temp.append(pos)
                neg_temp.append(neg)
            pos_temps.append(np.hstack(pos_temp))
            neg_temp = np.hstack(neg_temp)

        pos_mask = []
        for i in range(self.num_crops - 1):
            pos_mask.append(torch.from_numpy(pos_temps[1 + i]))
        neg_mask = torch.from_numpy(neg_temp - sum(pos_temps))

        if self.use_gpu:
            for i in range(len(pos_mask)):
                pos_mask[i] = pos_mask[i].cuda(non_blocking=True)
            neg_mask = neg_mask.cuda(non_blocking=True)

        self.pos_mask, self.neg_mask = pos_mask, neg_mask

    def forward(self, embedding: torch.Tensor):
        """
        Calculate the loss. Operates on embeddings tensor.
        """
        assert embedding.ndim == 2
        assert embedding.shape[1] == int(self.buffer_params.embedding_dim)

        batch_size = embedding.shape[0]
        T, num_crops = self.temperature, self.num_crops
        assert (
            batch_size % num_crops == 0
        ), "Batch size should be divisible by num_crops"

        # Step 1: gather all the embeddings. Shape example: 4096 x 128
        embeddings_buffer = self.gather_embeddings(embedding)

        # Step 2: matrix multiply: 64 x 128 with 4096 x 128 = 64 x 4096 and
        # divide by temperature.
        similarity = torch.exp(torch.mm(embedding, embeddings_buffer.t()) / T)
        Z, loss = 0.0, 0.0
        for loss_id in range(len(self.pos_mask)):
            pos = torch.sum(similarity * self.pos_mask[loss_id], 1)
            neg = torch.sum(similarity * self.neg_mask, 1)
            idx = (1 - torch.sum(self.pos_mask[loss_id], 1) > 0).detach()
            term_prob = pos / (pos + neg)
            term_prob[idx] = 1.0
            term_loss = torch.log(term_prob)
            Z += torch.sum(~idx).detach()
            loss -= torch.sum(term_loss)
        loss /= Z
        return loss

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "temperature": self.temperature,
            "dist_rank": self.dist_rank,
        }
        return pprint.pformat(repr_dict, indent=2)
