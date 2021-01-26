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
    """
    VISSL wraps around transforms so that they work with the multimodal input.
    VISSL supports batches that come from several datasets and sources. Hence
    the input batch (images, labels) always is a list.

    To apply the user defined transforms, VISSL takes "indices" as input which
    defines on what dataset/source data in the sample should the transform be
    applied to. For example:
        Assuming input sample is {
            "data": [dataset1_imgX, dataset2_imgY],
            "label": [dataset1_lblX, dataset2_lblY]
        }
        and the transform is:
            TRANSFORMS:
                - name: RandomGrayscale
                  p: 0.2
                  indices: 0
        then the transform is applied only on dataset1_imgX. If however, the
        indices are either not specified or set to 0, 1 then the transform
        is applied on both dataset1_imgX and dataset2_imgY

    Since this structure of data is introduced by vissl, the SSLTransformsWrapper
    takes care of dealing with the multi-modality input by wrapping the
    original transforms (pytorch transforms or custom transforms defined by user)
    and calling each transform on each index.

    VISSL also supports _TRANSFORMS_WITH_LABELS transforms that modify the label
    or are used to generate the labels used in self-supervised learning tasks like
    Jigsaw. When the transforms in _TRANSFORMS_WITH_LABELS are called, the new
    label is also returned besides the transformed image.

    VISSL also supports the _TRANSFORMS_WITH_COPIES which are transforms
    that basically generate several copies of image. Common example
    of self-supervised training methods that do this is SimCLR, SwAV, MoCo etc
    When a transform from _TRANSFORMS_WITH_COPIES is used, the SSLTransformsWrapper
    will flatten the transform output.
    For example for the input [img1], if we apply ImgReplicatePil to replicate
    the image 2 times:
        SSLTransformsWrapper(
            ImgReplicatePil(num_times=2), [img1]
        )
        will output [img1_1, img1_2] instead of nested list [[img1_1, img1_2]].
    The benefit of this is that the next set of transforms specified by user can now
    operate on img1_1 and img1_2 as the input becomes multi-modal nature.

    VISSL also supports _TRANSFORMS_WITH_GROUPING which essentially means
    that a single transform should be applied on the full multi-modal input
    together instead of separately. This is common transform used in BYOL/
    For example:
        SSLTransformsWrapper(
            ImgPilMultiCropRandomApply(
                RandomApply, prob=[0.0, 0.2]
            ), [img1_1, img1_2]
        )
        this will apply RandomApply on img1_1 with prob=0.0 and on img1_2 with
        prob=0.2
    """

    def __init__(self, indices, **args):
        """
        Args:
            indices (List[int]) (Optional): the indices list on which transform should
                                 be applied for the input which is always a list
                                 Example: minibatch of size=2 looks like [[img1], [img2]]).
                                 If indices is not specified, transform is applied
                                 to all the multi-modal input.
            args (dict): the arguments that the transform takes

        """
        self.indices = set(indices)
        self.name = args["name"]
        self.transform = build_transform(args)

    def _is_transform_with_labels(self):
        """
        _TRANSFORMS_WITH_LABELS = ["ImgRotatePil", "ShuffleImgPatches"]
        """
        if self.name in _TRANSFORMS_WITH_LABELS:
            return True
        return False

    def _is_transform_with_copies(self):
        """
        _TRANSFORMS_WITH_COPIES = [
            "ImgReplicatePil",
            "ImgPilToPatchesAndImage",
            "ImgPilToMultiCrop",
        ]
        """
        if self.name in _TRANSFORMS_WITH_COPIES:
            return True
        return False

    def _is_grouping_transform(self):
        """
        _TRANSFORMS_WITH_GROUPING = ["ImgPilMultiCropRandomApply"]
        """
        if self.name in _TRANSFORMS_WITH_GROUPING:
            return True
        return False

    def __call__(self, sample):
        """
        Apply each transform on the specified indices of each entry in
        the input sample.
        """
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
    """
    Given the list of user specified transforms, return the
    torchvision.transforms.Compose() version of the transforms. Each transform
    in the composition is SSLTransformsWrapper which wraps the original
    transforms to handle multi-modal nature of input.
    """
    output_transforms = []
    for transform_config in input_transforms_list:
        transform = SSLTransformsWrapper.from_config(transform_config)
        output_transforms.append(transform)
    return pth_transforms.Compose(output_transforms)


FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "vissl.data.ssl_transforms")

__all__ = ["SSLTransformsWrapper", "get_transform"]
