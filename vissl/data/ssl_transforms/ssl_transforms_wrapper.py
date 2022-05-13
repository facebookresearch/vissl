# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from classy_vision.dataset.transforms import (
    build_transform as build_classy_transform,
    register_transform,
)
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from vissl.utils.misc import is_augly_available


if is_augly_available():
    import augly.image as imaugs  # NOQA

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
_TRANSFORMS_WITH_OVERWRITE_ENTIRE_BATCH = ["OneHotEncode"]

DEFAULT_TRANSFORM_TYPES = {
    "TRANSFORMS_WITH_LABELS": _TRANSFORMS_WITH_LABELS,
    "TRANSFORMS_WITH_COPIES": _TRANSFORMS_WITH_COPIES,
    "TRANSFORMS_WITH_GROUPING": _TRANSFORMS_WITH_GROUPING,
    "TRANSFORMS_WITH_OVERWRITE_ENTIRE_BATCH": _TRANSFORMS_WITH_OVERWRITE_ENTIRE_BATCH,
}


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

    def __init__(
        self, indices, transform_types, transform_receives_entire_batch=False, **args
    ):
        """
        Args:
            indices (List[int]) (Optional): the indices list on which transform should
                                 be applied for the input which is always a list
                                 Example: minibatch of size=2 looks like [[img1], [img2]]).
                                 If indices is not specified, transform is applied
                                 to all the multi-modal input.
            args (dict): the arguments that the transform takes
            transform_types (dict): Types of transforms.
            transform_receives_entire_batch (bool): Whether or not the transforms receive the
                                           entire batch, or just one img.

        """
        self.indices = set(indices)
        self.name = args["name"]
        self.transform = self._build_transform(args)
        self.transform_receives_entire_batch = transform_receives_entire_batch
        self.transforms_with_labels = transform_types["TRANSFORMS_WITH_LABELS"]
        self.transforms_with_copies = transform_types["TRANSFORMS_WITH_COPIES"]
        self.transforms_with_overwrite_entire_batch = transform_types[
            "TRANSFORMS_WITH_OVERWRITE_ENTIRE_BATCH"
        ]
        self.transforms_with_grouping = transform_types["TRANSFORMS_WITH_GROUPING"]

    def _build_transform(self, args):
        if "transform_type" not in args:
            # Default to classy transform.
            return build_classy_transform(args)
        elif args["transform_type"] == "augly":
            # Build augly transform.
            return self._build_augly_transform(args)
        else:
            raise RuntimeError(
                f"Transform type: { args.transform_type } is not supported"
            )

    def _build_augly_transform(self, args):
        assert is_augly_available(), "Please pip install augly."

        # the name should be available in augly.image
        # if users specify the transform name in snake case,
        # we need to convert it to title case.
        name = args["name"]

        if not hasattr(imaugs, name):
            # Try converting name to title case.
            name = name.title().replace("_", "")

        assert hasattr(imaugs, name), f"{name} isn't a registered tranform for augly."

        # Delete superfluous keys.
        del args["name"]
        del args["transform_type"]

        return getattr(imaugs, name)(**args)

    def _is_transform_with_labels(self):
        """
        _TRANSFORMS_WITH_LABELS = ["ImgRotatePil", "ShuffleImgPatches"]
        """
        return self.name in self.transforms_with_labels

    def _is_transform_with_copies(self):
        """
        _TRANSFORMS_WITH_COPIES = [
            "ImgReplicatePil",
            "ImgPilToPatchesAndImage",
            "ImgPilToMultiCrop",
        ]
        """
        return self.name in self.transforms_with_copies

    def _is_overwrite_entire_batch_transform(self):
        """
        _TRANSFORMS_WITH_OVERWRITE_ENTIRE_BATCH = []
        """
        return self.name in self.transforms_with_overwrite_entire_batch

    def _is_grouping_transform(self):
        """
        _TRANSFORMS_WITH_GROUPING = ["ImgPilMultiCropRandomApply"]
        """
        return self.name in self.transforms_with_grouping

    def __call__(self, batch_or_img):
        """
        Apply each transform on the specified indices of each entry in
        the input sample or batch.
        """
        if self._is_overwrite_entire_batch_transform():
            # Transform the entire batch. This is currently only used for hive dataset.
            batch_or_img = self.transform(batch_or_img)
        elif self.transform_receives_entire_batch:
            # Transform every sample of the batch. This is currently only used for hive dataset.
            for i, sample in enumerate(batch_or_img):
                batch_or_img[i] = self._transform(sample)
        else:
            # Transform the sample. This is the default behavior.
            batch_or_img = self._transform(batch_or_img)

        return batch_or_img

    def _transform(self, sample):
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
                # Generalized transformation: we can add additional
                # keys inside the sample (for instance a label, a mask, etc)
                if isinstance(output, dict):
                    sample["data"][idx] = output["data"]
                    for k, v in output.items():
                        if k != "data":
                            sample.setdefault(k, []).append(v)
                # Deprecated generalized transformation (only working
                # for "labels"). Return a map instead
                elif self._is_transform_with_labels():
                    sample["data"][idx] = output[0]
                    sample["label"][-1] = output[1]
                # Transformation on the data only
                else:
                    sample["data"][idx] = output

        if self._is_transform_with_copies():
            # if the transform makes copies of the data, we just flatten the list
            # so the next set of transforms will operate on more indices. We also
            # need to replicate the label and metadata.

            data, label, data_valid, data_idx = [], [], [], []
            for i, img_replicas in enumerate(sample["data"]):
                num_times = len(img_replicas)
                data += img_replicas
                label += [sample["label"][i]] * num_times
                data_valid += [sample["data_valid"][i]] * num_times
                data_idx += [sample["data_idx"][i]] * num_times

            sample["data"] = data
            sample["label"] = label
            sample["data_idx"] = data_idx
            sample["data_valid"] = data_valid

        return sample

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        transform_types: Dict[str, Any],
        transform_receives_entire_batch=False,
    ) -> "SSLTransformsWrapper":
        indices = config.get("indices", [])
        return cls(
            indices,
            transform_types=transform_types,
            transform_receives_entire_batch=transform_receives_entire_batch,
            **config,
        )
