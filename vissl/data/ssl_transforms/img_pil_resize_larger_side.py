# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("ImgPilResizeLargerSide")
class ImgPilResizeLargerSide(ClassyTransform):
    def __init__(
        self,
        size: int,
    ):
        """
        Resizes the larger side to "size". Note that the torchvision.Resize transform
        crops the smaller edge to the provided size and is unable to crop the larger
        side. This is common in the copy detection and instance retrieval literature.
        """
        self.size = size

    def __call__(self, img):
        # Resize the longest side to self.size.
        img_size_hw = np.array((img.size[1], img.size[0]))
        ratio = float(self.size) / np.max(img_size_hw)
        new_size = tuple(np.round(img_size_hw * ratio).astype(np.int32))
        img_resized = img.resize((new_size[1], new_size[0]), Image.BILINEAR)

        return img_resized

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilResizeLargerSide":
        """
        Instantiates ImgPilRandomSolarize from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilRandomSolarize instance.
        """
        size = config.get("size", 1024)

        return cls(size)
