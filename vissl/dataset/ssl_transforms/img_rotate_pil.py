#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import Any, Dict

import torch
import torchvision.transforms.functional as TF
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgRotatePil")
class ImgRotatePil(ClassyTransform):
    def __init__(self, num_angles=4, num_rotations_per_img=1):
        self.num_angles = num_angles
        self.num_rotations_per_img = num_rotations_per_img
        # the last angle is 360 and 1st angle is 0, both give the original image.
        # 360 is not useful so remove it
        self.angles = torch.linspace(0, 360, num_angles + 1)[:-1]

    def __call__(self, image):
        label = torch.randint(self.num_angles, [1]).item()
        img = TF.rotate(image, self.angles[label])
        return img, label

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgRotatePil":
        num_angles = config.get("num_angles", 4)
        num_rotations_per_img = config.get("num_rotations_per_img", 1)
        assert num_rotations_per_img == 1, "Only num_rotations_per_img=1 allowed"
        logging.info(f"ImgRotatePil | Using num_angles: {num_angles}")
        logging.info(
            f"ImgRotatePil | Using num_rotations_per_img: {num_rotations_per_img}"
        )
        return cls(num_angles=num_angles, num_rotations_per_img=num_rotations_per_img)
