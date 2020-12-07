# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""
Code modified from https://github.com/google-research/vision_transformer
and https://www.internalfb.com/D24577730, as per https://arxiv.org/abs/2010.11929
"""

import logging
from typing import List

import torch
import torch.nn as nn
from vissl.models.trunks import register_model_trunk
from vissl.utils.hydra_config import AttrDict
from classy_vision.models import VisionTransformer as ClassyViT, build_model


@register_model_trunk("vision_transformer")
class VisionTransformer(nn.Module):
    """
    Wrapper for ClassVision vision_transformer model
    """
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()

        # TODO: This vision transformer implementation was intended to mirror
        #  how e.g. RegNet is implemented: basically being a wrapper for the
        #  ClassyVision implementation. However, the ClassyVision ViT
        #  implementation performs a number of operations in its .forward()
        #  method that are not contained in modules. This appears unlike the
        #  RegNet (and other model?) implementations, in which a forward
        #  pass can be achieved by calling each module in succession (usually
        #  by calling  vissl.models.model_helpers.get_trunk_forward_outputs()).
        #  Certain features  seem to rely on the model being fully
        #  modularized in this way, including activation checkpointing and
        #  access to intermediate representations.
        # self.use_activation_checkpointing = (
        #     model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        # )
        # self.activation_checkpointing_splits = (
        #     model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        # )
        #
        # if self.use_activation_checkpointing:
        #     logging.info(
        #         f"Activation checkpointing in use. {self.activation_checkpointing_splits} chunks"
        #     )

        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
        trunk_config = model_config.TRUNK.TRUNK_PARAMS.VISION_TRANSFORMERS

        if "name" in trunk_config:
            name = trunk_config["name"]
            logging.info(f"Building model: Vision Transformer: {name}")
            model = build_model({"name": name})
        else:
            logging.info("Building model: Vision Transformer from yaml config")
            model = ClassyViT.from_config(trunk_config)

        # TODO: This is where we would collect the model's modules into a
        #  list
        # feature_blocks: List[Tuple[str, nn.Module]] = []
        # for k,v in model.named_children():
        #     feature_blocks.append((k, v))

        # TODO: This may be a hacky workaround
        self.model = model

    def forward(self, x: torch.Tensor, out_feat_keys: List[str] = None
                ) -> List[torch.Tensor]:
        # TODO: Passing out_feat_keys is currently unsupported for reasons
        #  described above.
        x = self.model(x)
        # TODO: Unsqueeze is because models.base_ssl_model.heads_forward()
        #  assumes dimension 0 is feature dimension. Is there somewhere else
        #  the unsqueezing of dimension 0 should be handled?
        x = x.unsqueeze(0)
        return x

