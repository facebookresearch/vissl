# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from io import BytesIO

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("TensorToImgPil")
class TensorToImgPil(ClassyTransform):
    """
    Tensor to Pil Image. Currently Used for OnBoxDataLoader, as images are
    returned as tensors, but other transforms expect an Image PIL.
    """

    def __call__(self, sample):
        # Convert tensors to image pil for subsequent transforms.
        for i, tensor in enumerate(sample["data"]):
            sample["data"][i] = self._tensor_to_img_pil(tensor)

        sample = self._reformat_sample(sample)

        return sample

    def _tensor_to_img_pil(self, tensor):
        data = tensor.numpy().tobytes()
        with Image.open(BytesIO(data), mode="r") as img:
            image = img.convert("RGB")

        return image

    def _reformat_sample(self, sample):
        # Reformat label and data_valid to expected format.
        label, data_valid = sample["label"]
        label, data_valid = label.tolist(), data_valid.int().tolist()
        sample["data_idx"] = sample["id"]
        data_idx, _ = sample["id"]
        data_idx = data_idx.tolist()

        sample["label"] = label
        sample["data_valid"] = data_valid
        sample["data_idx"] = data_idx

        return sample
