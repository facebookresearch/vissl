# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random
from typing import Any, Dict

import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import ImageFilter


@register_transform("ImgPilGaussianBlur")
class ImgPilGaussianBlur(ClassyTransform):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.

    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p, radius_min, radius_max):
        """
        Args:
            p (float): probability of applying gaussian blur to the image
            radius_min (float): blur kernel minimum radius used by ImageFilter.GaussianBlur
            radius_max (float): blur kernel maximum radius used by ImageFilter.GaussianBlur
        """
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        should_blur = np.random.rand() <= self.prob
        if not should_blur:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilGaussianBlur":
        """
        Instantiates ImgPilGaussianBlur from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilGaussianBlur instance.
        """
        prob = config.get("p", 0.5)
        radius_min = config.get("radius_min", 0.1)
        radius_max = config.get("radius_max", 2.0)
        return cls(p=prob, radius_min=radius_min, radius_max=radius_max)
