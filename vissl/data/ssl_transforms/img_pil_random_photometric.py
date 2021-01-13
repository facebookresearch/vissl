# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from vissl.data.ssl_transforms.pil_photometric_transforms_lib import (
    AutoContrastTransform,
    RandomPosterizeTransform,
    RandomSharpnessTransform,
    RandomSolarizeTransform,
)


@register_transform("ImgPilRandomPhotometric")
class ImgPilRandomPhotometric(ClassyTransform):
    """
    Randomly apply some photometric transforms to an image.
    This was used in PIRL - https://arxiv.org/abs/1912.01991
    """

    def __init__(self, p):
        """
        Inputs
        - p (float): Probability of applying the transforms
        """
        assert isinstance(p, float), f"p must be a float value. Found {type(p)}"
        assert p >= 0 and p <= 1
        transforms = [
            RandomPosterizeTransform(),
            RandomSharpnessTransform(),
            RandomSolarizeTransform(),
            AutoContrastTransform(),
        ]
        self.transform = pth_transforms.RandomApply(transforms, p)
        logging.info(f"ImgPilRandomPhotometric with prob {p} and {transforms}")

    def __call__(self, image):
        return self.transform(image)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilRandomPhotometric":
        p = config.get("p", 0.66)
        return cls(p=p)
