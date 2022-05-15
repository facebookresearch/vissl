# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Union

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

    def __call__(
        self, data: Union[List[Dict[str, List[List[int]]]], Dict[str, List[List[int]]]]
    ):
        """
        Args:
            data (Union[List[Dict[str, List[List[int]]]], Dict[str, List[List[int]]]]). E.g:
                data = {"label": [[0]]}
        """
        # Handle case where data is a single example or an entire batch.
        data_list = data if isinstance(data, list) else [data]

        # One hot encode the labels.
        for sample in data_list:
            for idx, labels in enumerate(sample["label"]):
                sample["label"][idx] = self._one_hot_encode(torch.tensor(labels))

        return data

    def _one_hot_encode(self, labels):
        # Create tensor of dim (num_classes).
        one_hot_encoded = torch.zeros(
            (self.num_classes), dtype=torch.long, device=labels.device
        )

        # Change 0 to 1 for all labels.
        one_hot_encoded = one_hot_encoded.scatter_(0, labels, 1)

        return one_hot_encoded
