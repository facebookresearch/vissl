#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import Any, Dict

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgReplicatePil")
class ImgReplicatePil(ClassyTransform):
    """
    Adds the same image multiple times to the batch K times so that the batch.
    Size is now N*K. Use the flatten_collator to convert into batches.
    """

    def __init__(self, num_times):
        self.num_times = num_times

    def __call__(self, image):
        output = []
        for _ in range(self.num_times):
            output.append(image.copy())
        return output

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgReplicatePil":
        num_times = config.get("num_times", 2)
        assert num_times > 0, "num_times should be positive"
        logging.info(f"ImgReplicatePil | Using num_times: {num_times}")
        return cls(num_times=num_times)
