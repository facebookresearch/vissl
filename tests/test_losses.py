#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Minimal testing of the losses: ensure that a forward pass with believable dimensions succeeds.
This does not make them correct per say.
"""

import unittest
from collections import namedtuple

import torch
from classy_vision.generic.distributed_util import set_cpu_device
from vissl.ssl_criterions.simclr_info_nce_loss import SimclrInfoNCECriterion
from vissl.ssl_criterions.swav_loss import SwAVCriterion


set_cpu_device()

BATCH_SIZE = 4096
EMBEDDING_DIM = 128
BUFFER_PARAMS_STRUCT = namedtuple(
    "BUFFER_PARAMS_STRUCT", ["EFFECTIVE_BATCH_SIZE", "WORLD_SIZE", "EMBEDDING_DIM"]
)
BUFFER_PARAMS = BUFFER_PARAMS_STRUCT(BATCH_SIZE, 1, EMBEDDING_DIM)


class TaskTest(unittest.TestCase):
    @staticmethod
    def _get_embedding():
        return torch.ones([BATCH_SIZE, EMBEDDING_DIM])

    def test_simclr_info_nce_loss(self):
        loss_layer = SimclrInfoNCECriterion(
            buffer_params=BUFFER_PARAMS, temperature=0.1, num_pos=1
        )
        _ = loss_layer(self._get_embedding())

    def test_swav_loss(self):
        loss_layer = SwAVCriterion(
            temperature=0.1,
            crops_for_assign=[0, 1],
            nmb_crops=2,
            nmb_iters=3,
            epsilon=0.05,
            use_double_prec=False,
            nmb_prototypes=[3000],
            local_queue_length=0,
            embedding_dim=EMBEDDING_DIM,
        )
        _ = loss_layer(scores=self._get_embedding(), head_id=0)
