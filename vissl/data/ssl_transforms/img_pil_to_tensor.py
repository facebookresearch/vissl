# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


try:
    import accimage
except ImportError:
    accimage = None


# See /pytorch/vision/torchvision/transforms/functional.py
def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


@register_transform("ImgToTensor")
class ImgToTensor(ClassyTransform):
    """
    The Transform that overrides the PyTorch transform to provide
    better transformation speed.

    # credits: mannatsingh@fb.com
    """

    def __call__(self, img: Image):
        assert _is_numpy(img) or _is_pil_image(img)

        arr = np.asarray(img)
        arr = np.moveaxis(arr, -1, 0)  # HWC to CHW format
        arr = arr.astype(np.float32) / 255
        return torch.from_numpy(arr)
