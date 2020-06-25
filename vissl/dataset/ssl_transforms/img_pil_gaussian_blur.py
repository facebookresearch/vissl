#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any, Dict

import cv2
import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("ImgPilGaussianBlur")
class ImgPilGaussianBlur(ClassyTransform):
    """
    Apply Gaussian Blur to the PIL image. Take the radius, kernel size and
    probability of application as the parameter.

    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, kernel, p, radius_min, radius_max):
        self.kernel = kernel
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        should_blur = np.random.rand() > self.prob
        if not should_blur:
            return img
        # randomly sample sigma
        sigma = np.random.rand() * (self.radius_max - self.radius_min) + self.radius_min
        blurred = cv2.GaussianBlur(np.asarray(img), (self.kernel, self.kernel), sigma)
        return Image.fromarray(blurred)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilGaussianBlur":
        kernel = config.get("kernel", 23)
        prob = config.get("p", 0.5)
        radius_min = config.get("radius_min", 0.1)
        radius_max = config.get("radius_max", 2.0)
        return cls(kernel=kernel, p=prob, radius_min=radius_min, radius_max=radius_max)
