# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgPilColorDistortion")
class ImgPilColorDistortion(ClassyTransform):
    """
    Apply Random color distortions to the input image.
    There are multiple different ways of applying these distortions.
    This implementation follows SimCLR - https://arxiv.org/abs/2002.05709
    It randomly distorts the hue, saturation, brightness of an image and can
    randomly convert the image to grayscale.
    """

    def __init__(self, strength):
        """
        Args:
            strength (float): A number used to quantify the strength of the
                              color distortion.
        """
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
        return self.transforms(image)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilColorDistortion":
        """
        Instantiates ImgPilColorDistortion from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilColorDistortion instance.
        """
        strength = config.get("strength", 1.0)
        logging.info(f"ImgPilColorDistortion | Using strength: {strength}")
        return cls(strength=strength)
