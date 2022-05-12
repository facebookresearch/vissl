# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import numpy as np
import PIL
import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from vissl.data.ssl_transforms.img_patches_tensor import ImgPatchesFromTensor


@register_transform("ImgPilToPatchesAndImage")
class ImgPilToPatchesAndImage(ClassyTransform):
    """
    Convert an input PIL image to Patches and Image
    This transform was proposed in PIRL - https://arxiv.org/abs/1912.01991.

    Input:
        PIL Image
    Returns:
        list containing N+1 elements
            + zeroth element: a RandomResizedCrop of the image
            + remainder: N patches extracted uniformly from a RandomResizedCrop
    """

    def __init__(
        self,
        crop_scale_image=(0.08, 1.0),
        crop_size_image=224,
        crop_scale_patches=(0.6, 1.0),
        crop_size_patches=255,
        permute_patches=True,
        num_patches=9,
    ):
        """
        Args:
            crop_scale_image (tuple of floats): scale for RandomResizedCrop of image
            crop_size_image (int): size for RandomResizedCrop of image
            crop_scale_patches (tuple of floats): scale for RandomResizedCrop of patches
            crop_size_patches (int): size for RandomResizedCrop of patches
            permute_patches (bool): permute the patches in any order
            num_patches (int): number of patches to create. should be a square integer.
        """
        assert isinstance(num_patches, int)
        splits_per_side = np.sqrt(num_patches)
        assert (
            splits_per_side**2 == num_patches
        ), "Num patches must be a perfect square integer."
        self.num_patches = num_patches
        assert len(crop_scale_image) == 2
        assert len(crop_scale_patches) == 2

        self.crop_image_tx = pth_transforms.RandomResizedCrop(
            scale=crop_scale_image, size=crop_size_image
        )
        self.crop_patches_tx = pth_transforms.RandomResizedCrop(
            scale=crop_scale_patches, size=crop_size_patches
        )

        self.image_to_patch_tx = ImgPatchesFromTensor(num_patches=num_patches)

    def __call__(self, image):
        cropped_image = self.crop_image_tx(image)
        cropped_patch_image = self.crop_patches_tx(image)

        # image to patch accepts a tensor or array
        image_array = np.array(cropped_patch_image).transpose(2, 0, 1)
        patch_arrays = self.image_to_patch_tx(image_array)
        patches = [PIL.Image.fromarray(x.transpose(1, 2, 0)) for x in patch_arrays]

        # permute patches in any order
        perm_order = np.random.permutation(self.num_patches)
        patches = [patches[x] for x in perm_order]

        # make image the first member of the list
        patches.insert(0, cropped_image)
        return patches

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgPilToPatchesAndImage":
        """
        Instantiates ImgPilToPatchesAndImage from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPilToPatchesAndImage instance.
        """
        return cls(**config)
