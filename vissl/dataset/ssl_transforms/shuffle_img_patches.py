# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
from typing import Any, Dict

import numpy as np
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ShuffleImgPatches")
class ShuffleImgPatches(ClassyTransform):
    def __init__(self, perm_file):
        self.perm_file = perm_file
        assert os.path.exists(perm_file), f"Permutation file NOT found: {perm_file}"
        self.perms = np.load(perm_file)
        if np.min(self.perms) == 1:
            self.perms = self.perms - 1
        logging.info(f"Loaded perm: {self.perms.shape}")

    def __call__(self, input_patches):
        perm_index = np.random.randint(self.perms.shape[0])
        shuffled_patches = [input_patches[i] for i in self.perms[perm_index]]
        # num_towers x C x H x W
        input_data = torch.stack(shuffled_patches)
        out_label = torch.Tensor([perm_index]).long()
        return input_data, out_label

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ShuffleImgPatches":
        assert "perm_file" in config, "Please specify the perm_file"
        return cls(perm_file=config["perm_file"])
