# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, Dict

import numpy as np
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgPilToRawTensor")
class ImgPilToRawTensor(ClassyTransform):
    """
    Convert a PIL image to Raw Tensor if we don't want to apply the default
    division by 255 by torchvision.transforms.ToTensor()
    """

    def __init__(self):
        logging.info("Constructing ImgPilToRawTensor transform")

    def __call__(self, image):
        img = np.array(image)
        # Image is of shape H x W x C. Convert to C x H x W and then torch tensor
        # float.
        img_raw_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return img_raw_tensor

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilToRawTensor":
        """
        Instantiates ImgPilToRawTensor from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilToRawTensor instance.
        """
        return cls()
