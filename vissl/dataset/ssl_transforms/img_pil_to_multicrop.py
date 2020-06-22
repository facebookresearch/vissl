#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Dict

import numpy as np
import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgPilToMultiCrop")
class ImgPilToMultiCrop(ClassyTransform):
    """
    Convert a PIL image to Multi-resolution Crops
    Input
    - PIL Image
    Returns
    - list containing crops
    """

    def __init__(self, total_nmb_crops, size_crops, nmb_crops, crop_scales):

        assert np.sum(nmb_crops) == total_nmb_crops
        assert len(size_crops) == len(nmb_crops)
        assert len(size_crops) == len(crop_scales)

        trans = []
        for i, sc in enumerate(size_crops):
            trans.extend(
                [
                    pth_transforms.Compose(
                        [pth_transforms.RandomResizedCrop(sc, scale=crop_scales[i])]
                    )
                ]
                * nmb_crops[i]
            )

        self.transforms = trans

    def __call__(self, image):
        return list(map(lambda trans: trans(image), self.transforms))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilToMultiCrop":
        return cls(**config)
