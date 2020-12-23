# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from enum import Enum
from typing import List

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
from vissl.models.model_helpers import (
    Flatten,
    _get_norm,
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk
from vissl.utils.hydra_config import AttrDict


# For more depths, add the block config here
BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 24, 36, 3),
}


class SUPPORTED_DEPTHS(int, Enum):
    RN50 = 50
    RN101 = 101
    RN152 = 152
    RN200 = 200


class INPUT_CHANNEL(int, Enum):
    lab = 1
    bgr = 3
    rgb = 3


class SUPPORTED_L4_STRIDE(int, Enum):
    one = 1
    two = 2


@register_model_trunk("resnet")
class ResNeXt(nn.Module):
    """
    Wrapper for TorchVison ResNet Model to support different depth and
    width_multiplier. We provide flexibility with LAB input, stride in last
    ResNet block and type of norm (BatchNorm, LayerNorm)
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(ResNeXt, self).__init__()
        self.model_config = model_config
        logging.info(
            "ResNeXT trunk, supports activation checkpointing. {}".format(
                "Activated"
                if self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
                else "Deactivated"
            )
        )

        self.trunk_config = self.model_config.TRUNK.TRUNK_PARAMS.RESNETS
        self.depth = SUPPORTED_DEPTHS(self.trunk_config.DEPTH)
        self.width_multiplier = self.trunk_config.WIDTH_MULTIPLIER
        self._norm_layer = _get_norm(self.trunk_config.NORM)
        self.groups = self.trunk_config.GROUPS
        self.zero_init_residual = self.trunk_config.ZERO_INIT_RESIDUAL
        self.width_per_group = self.trunk_config.WIDTH_PER_GROUP
        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )

        (n1, n2, n3, n4) = BLOCK_CONFIG[self.depth]
        logging.info(
            f"Building model: ResNeXt"
            f"{self.depth}-{self.groups}x{self.width_per_group}d-"
            f"w{self.width_multiplier}-{self._norm_layer.__name__}"
        )

        model = models.resnet.ResNet(
            block=Bottleneck,
            layers=(n1, n2, n3, n4),
            zero_init_residual=self.zero_init_residual,
            groups=self.groups,
            width_per_group=self.width_per_group,
            norm_layer=self._norm_layer,
        )

        model.inplanes = 64 * self.width_multiplier
        dim_inner = 64 * self.width_multiplier
        # some tasks like Colorization https://arxiv.org/abs/1603.08511 take input
        # as L channel of an LAB image. In that case we change the input channel
        # and re-construct the conv1
        self.input_channels = INPUT_CHANNEL[self.model_config.INPUT_TYPE]

        model_conv1 = nn.Conv2d(
            self.input_channels,
            model.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model_bn1 = self._norm_layer(model.inplanes)
        model_relu1 = model.relu
        model_maxpool = model.maxpool
        model_avgpool = model.avgpool
        model_layer1 = model._make_layer(Bottleneck, dim_inner, n1)
        model_layer2 = model._make_layer(Bottleneck, dim_inner * 2, n2, stride=2)
        model_layer3 = model._make_layer(Bottleneck, dim_inner * 4, n3, stride=2)

        # For some models like Colorization https://arxiv.org/abs/1603.08511,
        # due to the higher spatial resolution desired for pixel wise task, we
        # support using a different stride. Currently, we know stride=1 and stride=2
        # behavior so support only those.
        safe_stride = SUPPORTED_L4_STRIDE(self.trunk_config.LAYER4_STRIDE)
        model_layer4 = model._make_layer(
            Bottleneck, dim_inner * 8, n4, stride=safe_stride
        )

        # we mapped the layers of resnet model into feature blocks to facilitate
        # feature extraction at various layers of the model. The layers for which
        # to extract features is controlled by requested_feat_keys argument in the
        # forward() call.
        self._feature_blocks = nn.ModuleDict(
            [
                ("conv1", model_conv1),
                ("bn1", model_bn1),
                ("conv1_relu", model_relu1),
                ("maxpool", model_maxpool),
                ("layer1", model_layer1),
                ("layer2", model_layer2),
                ("layer3", model_layer3),
                ("layer4", model_layer4),
                ("avgpool", model_avgpool),
                ("flatten", Flatten(1)),
            ]
        )

        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
            "conv1": "conv1_relu",
            "res1": "maxpool",
            "res2": "layer1",
            "res3": "layer2",
            "res4": "layer3",
            "res5": "layer4",
            "res5avg": "avgpool",
            "flatten": "flatten",
        }

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        feat = transform_model_input_data_type(x, self.model_config)
        return get_trunk_forward_outputs(
            feat,
            out_feat_keys=out_feat_keys,
            feature_blocks=self._feature_blocks,
            feature_mapping=self.feat_eval_mapping,
            use_checkpointing=self.use_checkpointing,
            checkpointing_splits=self.num_checkpointing_splits,
        )
