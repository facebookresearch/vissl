#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from vissl.models.heads.linear_eval_mlp import LinearEvalMLP
from vissl.models.heads.mlp import MLP
from vissl.models.heads.siamese_concat_view import SiameseConcatView
from vissl.models.heads.swav_prototypes_head import SwAVPrototypesHead


HEADS = {
    "eval_mlp": LinearEvalMLP,
    "mlp": MLP,
    "siamese_concat_view": SiameseConcatView,
    "swav_head": SwAVPrototypesHead,
}
