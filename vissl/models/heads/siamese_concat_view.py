#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class SiameseConcatView(nn.Module):
    def __init__(self, model_config, num_towers):
        super().__init__()
        self.num_towers = num_towers

    def forward(self, batch):
        # batch dimension = (N * num_towers) x C x H x W
        siamese_batch_size = batch.shape[0]
        assert (
            siamese_batch_size % self.num_towers == 0
        ), f"{siamese_batch_size} not divisible by num_towers {self.num_towers}"
        batch_size = siamese_batch_size // self.num_towers
        out = batch.view(batch_size, -1)
        return out
