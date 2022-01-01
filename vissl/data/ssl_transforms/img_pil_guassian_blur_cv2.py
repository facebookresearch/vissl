# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import cv2
import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("ImgPilGaussianBlurCV2")
class ImgPilGaussianBlurCV2(ClassyTransform):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.

    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p, kernel_size, sigma_min=0.1, sigma_max=2.0):
        """
        Args:
            p (float): probability of applying gaussian blur to the image
            radius_min (float): blur kernel minimum radius used by ImageFilter.GaussianBlur
            radius_max (float): blur kernel maximum radius used by ImageFilter.GaussianBlur
        """
        self.prob = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        should_blur = np.random.rand() <= self.prob
        if not should_blur:
            return img

        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(
            np.array(img), (self.kernel_size, self.kernel_size), sigma
        )
        return Image.fromarray(img.astype(np.uint8))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilGaussianBlurCV2":
        """
        Instantiates ImgPilGaussianBlur from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilGaussianBlur instance.
        """
        prob = config.get("p", 0.5)

        sigma_min = config.get("sigma_min", 0.1)
        sigma_max = config.get("sigma_max", 2.0)
        kernel_size = config.get("kernel_size", 23)

        return cls(
            p=prob, kernel_size=kernel_size, sigma_min=sigma_min, sigma_max=sigma_max
        )
