# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
