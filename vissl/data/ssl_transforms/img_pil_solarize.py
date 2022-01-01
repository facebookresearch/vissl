# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import cv2
import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import ImageOps


@register_transform("ImgPilSolarize")
class ImgPilSolarize(ClassyTransform):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.

    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p, threshold=128):
        """
        Args:
            p (float): probability of applying gaussian blur to the image
            radius_min (float): blur kernel minimum radius used by ImageFilter.GaussianBlur
            radius_max (float): blur kernel maximum radius used by ImageFilter.GaussianBlur
        """
        self.prob = p
        self.threshold = threshold

    def __call__(self, img):
        should_blur = np.random.rand() <= self.prob
        if not should_blur:
            return img

        return ImageOps.solarize(img, self.threshold)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilSolarize":
        """
        Instantiates ImgPilGaussianBlur from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilGaussianBlur instance.
        """
        prob = config.get("p", 0.5)

        threshold = config.get("threshold", 128)

        return cls(p=prob, threshold=threshold)
