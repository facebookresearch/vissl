# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch


def _as_tensor(x: Tuple[int, int]) -> torch.Tensor:
    """
    An equivalent of `torch.as_tensor`, but works under tracing if input
    is a list of tensor. `torch.as_tensor` will record a constant in tracing,
    but this function will use `torch.stack` instead.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x)
    if isinstance(x, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in x):
        return torch.stack(x)
    return torch.as_tensor(x)


class MultiDimensionalTensor:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
    ):
        self.tensor = tensor
        self.mask = mask
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        """
        Effective batch size. For multi-crop augmentations,
        (as in SwAV https://arxiv.org/abs/2006.09882) this returns N * num_crops.
        Otherwise returns N.
        """
        return len(self.tensor)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C, H, W)
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @property
    def device(self):
        return self.tensor.device

    def to(self, device, non_blocking: bool):
        """
        Move the tensor and mask to the specified device.
        """
        # type: (Device) -> MultiDimensionalTensor # noqa
        cast_tensor = self.tensor.to(device, non_blocking=non_blocking)
        cast_mask = self.mask.to(device, non_blocking=non_blocking)
        return MultiDimensionalTensor(cast_tensor, cast_mask, self.image_sizes)

    @classmethod
    def from_tensors(cls, tensor_list: List[torch.Tensor]) -> "MultiDimensionalTensor":
        assert len(tensor_list) > 0
        assert isinstance(tensor_list, list)
        for t in tensor_list:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensor_list[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensor_list]
        image_sizes_tensor = [_as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values
        num_tensors = len(tensor_list)
        batch_shape_per_crop = list(tensor_list[0].shape[:-2]) + list(max_size)

        b, c, h, w = batch_shape_per_crop
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        nested_output_tensor = torch.zeros(
            (b * num_tensors, c, h, w), dtype=dtype, device=device
        )
        mask = torch.ones((b * num_tensors, h, w), dtype=torch.bool, device=device)

        for crop_num in range(num_tensors):
            img = tensor_list[crop_num]
            nested_output_tensor[
                (crop_num * b) : (crop_num + 1) * b, :, : img.shape[2], : img.shape[3]
            ].copy_(img)
            mask[
                (crop_num * b) : (crop_num + 1) * b, : img.shape[2], : img.shape[3]
            ] = False
        return MultiDimensionalTensor(nested_output_tensor, mask, image_sizes)
