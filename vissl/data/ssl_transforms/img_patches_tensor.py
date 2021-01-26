# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import math
from typing import Any, Dict

import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgPatchesFromTensor")
class ImgPatchesFromTensor(ClassyTransform):
    """
    Create image patches from a torch Tensor or numpy array.
    This transform was proposed in Jigsaw - https://arxiv.org/abs/1603.09246

    Args:
        num_patches (int): how many image patches to create
        patch_jitter (int): space to leave between patches
    """

    def __init__(self, num_patches=9, patch_jitter=21):
        self.num_patches = num_patches
        self.patch_jitter = patch_jitter
        assert self.patch_jitter > 0, "Negative jitter not supported"
        self.grid_side_len = int(math.sqrt(self.num_patches))  # usually = 3
        logging.info(
            f"ImgPatchesFromTensor: num_patches: {num_patches} "
            f"patch_jitter: {patch_jitter}"
        )

    def __call__(self, image):
        """
        Input image which is a torch.Tensor object of shape 3 x H x W
        """
        data = []
        grid_size = int(image.shape[1] / self.grid_side_len)
        patch_size = grid_size - self.patch_jitter
        jitter = np.random.randint(
            0, self.patch_jitter, (2, self.grid_side_len, self.grid_side_len)
        )

        for i in range(self.grid_side_len):
            for j in range(self.grid_side_len):
                x_offset = i * grid_size
                y_offset = j * grid_size
                grid_cell = image[
                    :, y_offset : y_offset + grid_size, x_offset : x_offset + grid_size
                ]

                patch = grid_cell[
                    :,
                    jitter[1, i, j] : jitter[1, i, j] + patch_size,
                    jitter[0, i, j] : jitter[0, i, j] + patch_size,
                ]
                assert patch.shape[1] == patch_size, "Image not cropped properly"
                assert patch.shape[2] == patch_size, "Image not cropped properly"
                # copy patch data so that all patches are different in underlying memory
                data.append(np.copy(patch))
        return data

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPatchesFromTensor":
        """
        Instantiates ImgPatchesFromTensor from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPatchesFromTensor instance.
        """
        num_patches = config.get("num_patches", 9)
        patch_jitter = config.get("patch_jitter", 21)
        logging.info(f"ImgPatchesFromTensor | Using num_patches: {num_patches}")
        logging.info(f"ImgPatchesFromTensor | Using patch_jitter: {patch_jitter}")
        return cls(num_patches=num_patches, patch_jitter=patch_jitter)
