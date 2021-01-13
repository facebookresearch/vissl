# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from classy_vision.models import RegNet as ClassyRegNet, build_model
from vissl.models.model_helpers import (
    Flatten,
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk
from vissl.utils.hydra_config import AttrDict


@register_model_trunk("regnet")
class RegNet(nn.Module):
    """
    Wrapper for ClassyVision RegNet model so we can map layers into feature
    blocks to facilitate feature extraction and benchmarking at several layers.

    This model is defined on the fly from a RegNet base class and a configuration file.

    We follow the feature naming convention defined in the ResNet vissl trunk.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.model_config = model_config
        self.use_activation_checkpointing = (
            model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.activation_checkpointing_splits = (
            model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )

        if self.use_activation_checkpointing:
            logging.info(
                f"Using Activation checkpointing. {self.activation_checkpointing_splits} chunks"
            )

        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
        trunk_config = model_config.TRUNK.TRUNK_PARAMS.REGNET

        if "name" in trunk_config:
            name = trunk_config["name"]
            logging.info(f"Building model: RegNet: {name}")
            model = build_model({"name": name})
        else:
            logging.info("Building model: RegNet from yaml config")
            model = ClassyRegNet.from_config(trunk_config)

        # Now map the models to the structure we want to expose for SSL tasks
        # The upstream RegNet model is made of :
        # - `stem`
        # - n x blocks in trunk_output, named `block1, block2, ..`

        # We're only interested in the stem and successive blocks
        # everything else is not picked up on purpose
        feature_blocks: List[Tuple[str, nn.Module]] = []

        # - get the stem
        feature_blocks.append(("conv1", model.stem))

        # - get all the feature blocks
        for k, v in model.trunk_output.named_children():
            assert k.startswith("block"), f"Unexpected layer name {k}"
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f"res{block_index}", v))

        # - finally, add avgpool and flatten.
        feature_blocks.append(("avgpool", nn.AdaptiveAvgPool2d((1, 1))))
        feature_blocks.append(("flatten", Flatten(1)))

        self._feature_blocks = nn.ModuleDict(feature_blocks)

    def forward(self, x, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        model_input = transform_model_input_data_type(x, self.model_config)
        return get_trunk_forward_outputs(
            feat=model_input,
            out_feat_keys=out_feat_keys,
            feature_blocks=self._feature_blocks,
            use_checkpointing=self.use_activation_checkpointing,
            checkpointing_splits=self.activation_checkpointing_splits,
        )
