# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Union

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("MNISTImgPil2RGB")
class MNISTImgPil2RGB(ClassyTransform):
    """
    Convert a PIL image to RGB mode.

    This transform is necessary to adapt datasets such as MNIST (which has 1 color channel)
    as input to traditional architectures (like ResNet50) requiring 3 color channels.

    We first create an image of the same size as the original (or fixed "size" if provided as input) and
    paste the original image in it. If a different size is provided, the "box" indicates where the paste
    of the original image will be in the target image.

    Default behavior (no arguments provided) is to keep the original image size.

    The output is a PIL image in RGB mode.
    """

    def __init__(self, size: Union[int, List[int]], box: List[int]):
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        if size:
            assert (
                len(size) == 2
            ), f'ImgPil2RGB: Expected "size" parameter of length 2, got: {size}'
        assert (
            len(box) == 2
        ), f'ImgPil2RGB: Expected "box" parameter of length 2, got: {box}'
        self.size = size
        self.box = box

    def __call__(self, original_img: Image.Image) -> Image.Image:
        if self.size:
            img = Image.new(mode="RGB", size=self.size)
            img.paste(original_img, box=self.box)
        else:
            img = Image.new(mode="RGB", size=original_img.size)
            img.paste(original_img)
        return img

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MNISTImgPil2RGB":
        """
        Instantiates ImgPil2LabTensor from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgPil2RGB instance.
        """
        size = config.get("size", [])
        box = config.get("box", [0, 0])
        return cls(size=size, box=box)
