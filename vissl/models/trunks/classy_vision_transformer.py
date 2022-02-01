# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import List

import torch
import torch.nn as nn
from classy_vision.models import VisionTransformer as ClassyVisionTransformer
from vissl.config import AttrDict
from vissl.models.trunks import register_model_trunk


@register_model_trunk("classy_vit")
class ClassyViT(nn.Module):
    """
    Simple wrapper for ClassyVision Vision Transformer model.
    This model is defined on the fly from a Vision Transformer base class and
    a configuration file.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.model_config = model_config

        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
        trunk_config = copy.deepcopy(model_config.TRUNK.VISION_TRANSFORMERS)

        logging.info("Building model: Vision Transformer from yaml config")
        trunk_config = AttrDict({k.lower(): v for k, v in trunk_config.items()})

        self.model = ClassyVisionTransformer(
            image_size=trunk_config.image_size,
            patch_size=trunk_config.patch_size,
            num_layers=trunk_config.num_layers,
            num_heads=trunk_config.num_heads,
            hidden_dim=trunk_config.hidden_dim,
            mlp_dim=trunk_config.mlp_dim,
            dropout_rate=trunk_config.dropout_rate,
            attention_dropout_rate=trunk_config.attention_dropout_rate,
            classifier=trunk_config.classifier,
        )

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        x = self.model(x)
        x = x.unsqueeze(0)
        return x
