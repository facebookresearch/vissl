# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

    def __init__(
        self,
        strength: float = 1.0,
        brightness: float = 0.8,
        contrast: float = 0.8,
        saturation: float = 0.8,
        hue: float = 0.2,
    ):
        """
        Args:
            strength (float): quantify the strength of the color distortion
            brightness (float): default brightness then multiplied by strength
            contrast (float): default contrast then multiplied by strength
            saturation (float): default saturation then multiplied by strength
            hue (float): default hue then multiplied by strength
        """
        self.strength = strength
        self.color_jitter = pth_transforms.ColorJitter(
            brightness=brightness * self.strength,
            contrast=contrast * self.strength,
            saturation=saturation * self.strength,
            hue=hue * self.strength,
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
