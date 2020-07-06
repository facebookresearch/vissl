# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import pprint

import numpy as np
import torch
from classy_vision.losses import register_loss
from vissl.ssl_criterions.simclr_info_nce_loss import (
    DistributedSimclrInfoNCELoss,
    SimclrInfoNCECriterion,
)


@register_loss("multicrop_simclr_info_nce_loss")
class DistributedMultiCropSimclrInfoNCELoss(DistributedSimclrInfoNCELoss):
    def __init__(self, config, device: str = "gpu"):
        super(DistributedSimclrInfoNCELoss, self).__init__()

        self.loss_config = config.SIMCLR_INFO_NCE_LOSS
        # loss constants
        self.temperature = self.loss_config.TEMPERATURE
        self.buffer_params = self.loss_config.BUFFER_PARAMS
        self.multi_crop_params = self.loss_config.MULTI_CROP_PARAMS
        self.nmb_crops = self.multi_crop_params.NMB_CROPS
        self.info_criterion = MultiCropSimclrInfoNCECriterion(
            self.buffer_params, self.temperature, self.nmb_crops
        )

    def __repr__(self):
        repr_dict = {
            "name": self._get_name(),
            "info_average": self.info_criterion,
            "multi_crop": self.multi_crop_params,
        }
        return pprint.pformat(repr_dict, indent=2)


class MultiCropSimclrInfoNCECriterion(SimclrInfoNCECriterion):
    def __init__(self, buffer_params, temperature: float, nmb_crops: int):

        self.nmb_crops = nmb_crops
        super(MultiCropSimclrInfoNCECriterion, self).__init__(
            buffer_params, temperature
        )
        logging.info(f"Setting multicrop nmb_crops: {nmb_crops}")

    def precompute_pos_neg_mask(self):
        # computed once at the beginning of training
        total_images = self.buffer_params.EFFECTIVE_BATCH_SIZE
        world_size = self.buffer_params.WORLD_SIZE
        local_orig_images = total_images // world_size
        local_crops = local_orig_images * self.nmb_crops
        rank = self.dist_rank

        pos_temps = []
        for d in np.arange(self.nmb_crops):
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
        for i in range(self.nmb_crops - 1):
            pos_mask.append(torch.from_numpy(pos_temps[1 + i]))
        neg_mask = torch.from_numpy(neg_temp - sum(pos_temps))

        if self.use_gpu:
            for i in range(len(pos_mask)):
                pos_mask[i] = pos_mask[i].cuda(non_blocking=True)
            neg_mask = neg_mask.cuda(non_blocking=True)

        self.pos_mask, self.neg_mask = pos_mask, neg_mask

    def forward(self, embedding):
        assert embedding.ndim == 2
        assert embedding.shape[1] == int(self.buffer_params.EMBEDDING_DIM)

        batch_size = embedding.shape[0]
        T, nmb_crops = self.temperature, self.nmb_crops
        assert (
            batch_size % nmb_crops == 0
        ), "Batch size should be divisible by nmb_crops"

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
