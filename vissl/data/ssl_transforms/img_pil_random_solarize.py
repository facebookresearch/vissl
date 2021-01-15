# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from vissl.data.ssl_transforms.pil_photometric_transforms_lib import (
    RandomSolarizeTransform,
)


@register_transform("ImgPilRandomSolarize")
class ImgPilRandomSolarize(ClassyTransform):
    """
    Randomly apply solarization transform to an image.
    This was used in BYOL - https://arxiv.org/abs/2006.07733
    """

    def __init__(self, prob: float):
        """
        Args:
            p (float): Probability of applying the transform
        """
        self.p = prob
        transforms = [RandomSolarizeTransform()]
        self.transform = pth_transforms.RandomApply(transforms, self.p)
        logging.info(f"ImgPilRandomSolarize with prob {self.p} and {transforms}")

    def __call__(self, image):
        return self.transform(image)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilRandomSolarize":
        """
        Instantiates ImgPilRandomSolarize from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilRandomSolarize instance.
        """
        prob = config.get("p", 0.66)
        assert isinstance(prob, float), f"p must be a float value. Found {type(prob)}"
        assert prob >= 0 and prob <= 1
        return cls(prob=prob)
