# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgPilRandomColorJitter")
class ImgPilRandomColorJitter(ClassyTransform):
    """
    Apply Random color jitter to the input image.
    It randomly distorts the hue, saturation, brightness of an image.
    """

    def __init__(self, strength, prob):
        """
        Args:
            strength (float): A number used to quantify the strength of the color distortion.
            p (float): probability of random application
        """
        self.strength = strength
        self.p = prob
        self.color_jitter = pth_transforms.ColorJitter(
            0.8 * self.strength,
            0.8 * self.strength,
            0.8 * self.strength,
            0.2 * self.strength,
        )
        self.rnd_color_jitter = pth_transforms.RandomApply(
            [self.color_jitter], p=self.p
        )

    def __call__(self, image):
        return self.rnd_color_jitter(image)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilRandomColorJitter":
        """
        Instantiates ImgPilRandomColorJitter from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilRandomColorJitter instance.
        """
        strength = config.get("strength", 1.0)
        prob = config.get("p", 0.8)
        logging.info(f"ImgPilRandomColorJitter | Using strength: {strength}")
        return cls(strength=strength, prob=prob)
