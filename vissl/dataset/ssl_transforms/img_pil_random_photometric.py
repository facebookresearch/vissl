#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import Any, Dict

import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from vissl.dataset.ssl_transforms.pil_enhancements import (
    AutoContrastTransform,
    RandomPosterizeTransform,
    RandomSharpnessTransform,
    RandomSolarizeTransform,
)


@register_transform("ImgPilRandomPhotometric")
class ImgPilRandomPhotometric(ClassyTransform):
    def __init__(self, p):
        transforms = [
            RandomPosterizeTransform(),
            RandomSharpnessTransform(),
            RandomSolarizeTransform(),
            AutoContrastTransform(),
        ]
        self.transform = pth_transforms.RandomApply(transforms, p)
        logging.info(f"ImgPilRandomPhotometric with prob {p} and {transforms}")

    def __call__(self, image):
        transformed_image = self.transform(image)
        return transformed_image

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilRandomPhotometric":
        p = config.get("p", 0.66)
        return cls(p=p)
