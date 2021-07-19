# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Any

import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("OneHotEncode")
class OneHotEncode(ClassyTransform):
    """
    Prepare labels for multi-output support. That is when a single example has
    more than one label. Currently all that this transform does is one-hot encodes
    the labels. This is because the Classy accuracy meters already expects the labels
    to be one-hot encoded when in a multi-output mode.
    """

    def __init__(self, num_classes: int = 1000):
        """
        Args:
            num_classes (int): how many classes there are in total. Default 1000 for imagenet1k.
        """
        super().__init__()

        self.num_classes = num_classes

    def __call__(self, batch: Dict[str, Any]):
        """
        Args:
            batch (Dict[Str, Any]). Where batch["label"] is an array of tensors of ints.
            E.g. batch["label"] == [tensor([1, 2]), tensor([0])], means that the first
            example has labels 1 and 2 and second example has label 0.
        """
        # One hot encode the labels.
        for sample in batch:
            for idx, labels in enumerate(sample["label"]):
                sample["label"][idx] = self._one_hot_encode(labels)

        return batch

    def _one_hot_encode(self, labels):
        # Create tensor of dim (num_classes).
        one_hot_encoded = torch.zeros(
            (self.num_classes), dtype=torch.long, device=labels.device
        )

        # Change 0 to 1 for all labels.
        one_hot_encoded = one_hot_encoded.scatter_(0, labels, 1)

        return one_hot_encoded
