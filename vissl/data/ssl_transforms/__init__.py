# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pathlib import Path
from typing import Any, Dict

import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import build_transform, register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from classy_vision.generic.registry_utils import import_all_modules


# Below the transforms that require passing the labels as well. This is specifc
# to SSL only where we automatically generate the labels for training. All other
# transforms (including torchvision) require passing image only as input.
_TRANSFORMS_WITH_LABELS = ["ImgRotatePil", "ShuffleImgPatches"]
_TRANSFORMS_WITH_COPIES = [
    "ImgReplicatePil",
    "ImgPilToPatchesAndImage",
    "ImgPilToMultiCrop",
]
_TRANSFORMS_WITH_GROUPING = ["ImgPilMultiCropRandomApply"]


# we wrap around transforms so that they work with the multimodal input
@register_transform("SSLTransformsWrapper")
class SSLTransformsWrapper(ClassyTransform):
    def __init__(self, indices, **args):
        self.indices = set(indices)
        self.name = args["name"]
        self.transform = build_transform(args)

    def _is_transform_with_labels(self):
        if self.name in _TRANSFORMS_WITH_LABELS:
            return True
        return False

    def _is_transform_with_copies(self):
        if self.name in _TRANSFORMS_WITH_COPIES:
            return True
        return False

    def _is_grouping_transform(self):
        if self.name in _TRANSFORMS_WITH_GROUPING:
            return True
        return False

    def __call__(self, sample):
        # Run on all indices if empty set is passed.
        indices = self.indices if self.indices else set(range(len(sample["data"])))

        if self._is_grouping_transform():
            # if the transform needs to be applied to all the indices
            # together. For example: one might want to vary the intensity
            # of a transform across several crops of an image as in BYOL.
            output = self.transform(sample["data"])
            sample["data"] = output
        else:
            for idx in indices:
                output = self.transform(sample["data"][idx])
                if self._is_transform_with_labels():
                    sample["data"][idx] = output[0]
                    sample["label"].append(output[1])
                else:
                    sample["data"][idx] = output

        if self._is_transform_with_copies():
            # if the transform makes copies of the data, we just flatten the list
            # so the next set of transforms will operate on more indices
            sample["data"] = [val for sublist in sample["data"] for val in sublist]
            # now we replicate the rest of the metadata as well
            num_times = len(sample["data"])
            sample["label"] = sample["label"] * num_times
            sample["data_valid"] = sample["data_valid"] * num_times
            sample["data_idx"] = sample["data_idx"] * num_times
        return sample

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SSLTransformsWrapper":
        indices = config.get("indices", [])
        return cls(indices, **config)


def get_transform(input_transforms_list):
    output_transforms = []
    for transform_config in input_transforms_list:
        transform = SSLTransformsWrapper.from_config(transform_config)
        output_transforms.append(transform)
    return pth_transforms.Compose(output_transforms)


FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "vissl.data.ssl_transforms")

__all__ = ["SSLTransformsWrapper", "get_transform"]
