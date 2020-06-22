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


@register_transform("ImgPilColorDistortion")
class ImgPilColorDistortion(ClassyTransform):
    def __init__(self, strength):
        self.strength = strength
        self.color_jitter = pth_transforms.ColorJitter(
            0.8 * self.strength,
            0.8 * self.strength,
            0.8 * self.strength,
            0.2 * self.strength,
        )
        self.rnd_color_jitter = pth_transforms.RandomApply([self.color_jitter], p=0.8)
        self.rnd_gray = pth_transforms.RandomGrayscale(p=0.2)
        self.transforms = pth_transforms.Compose([self.rnd_color_jitter, self.rnd_gray])

    def __call__(self, image):
        transformed_image = self.transforms(image)
        return transformed_image

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilColorDistortion":
        strength = config.get("strength", 1.0)
        logging.info(f"ImgPilColorDistortion | Using strength: {strength}")
        return cls(strength=strength)
