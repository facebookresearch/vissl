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
        strength,
        brightness=0.8,
        contrast=0.8,
        saturation=0.8,
        hue=0.2,
        color_jitter_probability=0.8,
        grayscale_probability=0.2,
    ):
        """
        Args:
            strength (float): A number used to quantify the strength of the
                              color distortion.
            brightness (float): A floating point number used to quantify
                              jitter brightness. Default value is 0.8.
            contrast (float): A floating point number used to quantify
                              jitter contrast. Default value is 0.8.
            saturation (float):  A floating point number used to quantify
                              jitter saturation. Default value is 0.8.
            hue (float): A floating point number used to quantify
                              jitter hue. Default value is 0.2.
            color_jitter_probability (float): A floating point number used to
                            quantify to apply randomly a list of transformations
                            with a given probability. Default value is 0.8.
            grayscale_probability (float): A floating point number used to
                            quantify to apply randomly convert image to grayscale with
                            the assigned probability. Default value is 0.2.
        """
        self.strength = strength
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.color_jitter_probability = color_jitter_probability
        self.grayscale_probability = grayscale_probability
        self.color_jitter = pth_transforms.ColorJitter(
            self.brightness * self.strength,
            self.contrast * self.strength,
            self.saturation * self.strength,
            self.hue * self.strength,
        )
        self.rnd_color_jitter = pth_transforms.RandomApply(
            [self.color_jitter], p=self.color_jitter_probability
        )
        self.rnd_gray = pth_transforms.RandomGrayscale(p=self.grayscale_probability)
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
