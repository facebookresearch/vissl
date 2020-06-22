#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from vissl.models.trunks.alexnet_bvlc import AlexNetBvlc
from vissl.models.trunks.alexnet_colorization import AlexNetColorization
from vissl.models.trunks.alexnet_deepcluster import AlexNetDeepCluster
from vissl.models.trunks.alexnet_jigsaw import AlexNetJigsaw
from vissl.models.trunks.alexnet_rotnet import AlexNetRotNet
from vissl.models.trunks.efficientnet import EfficientNet
from vissl.models.trunks.resnext import ResNeXt


TRUNKS = {
    "alexnet_bvlc": AlexNetBvlc,
    "alexnet_colorization": AlexNetColorization,
    "alexnet_deepcluster": AlexNetDeepCluster,
    "alexnet_jigsaw": AlexNetJigsaw,
    "alexnet_rotnet": AlexNetRotNet,
    "resnet": ResNeXt,
    "efficientnet": EfficientNet,
}


__all__ = ["TRUNKS"]
