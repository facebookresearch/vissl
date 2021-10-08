# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
from typing import Any, Dict

import numpy as np
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from iopath.common.file_io import g_pathmgr
from vissl.utils.io import cache_url, is_url, load_file


@register_transform("ShuffleImgPatches")
class ShuffleImgPatches(ClassyTransform):
    """
    This transform is used to shuffle the list of tensors (usually
    image patches of shape C x H x W) according to a randomly selected
    permutation from a pre-defined set of permutations.

    This is a common operation used in Jigsaw approach https://arxiv.org/abs/1603.09246
    """

    def __init__(self, perm_file: str):
        """
        Args:
            perm_file (string): path to the file containing pre-defined permutations.
        """

        self.perm_file = perm_file
        self.perm_loaded = False
        self.perms = None

    def _load_perms(self):
        if is_url(self.perm_file):
            temp_cache_dir = tempfile.mkdtemp()
            cache_dir = os.path.join(temp_cache_dir, "perm_file_cache")
            cached_url_path = cache_url(url=self.perm_file, cache_dir=cache_dir)
            self.perm_file = cached_url_path
        assert g_pathmgr.exists(
            self.perm_file
        ), f"Permutation file NOT found: {self.perm_file}"
        logging.info(f"Loading permutation: {self.perm_file}")
        self.perms = load_file(self.perm_file)
        if np.min(self.perms) == 1:
            self.perms = self.perms - 1
        logging.info(f"Loaded perm: {self.perms.shape}")
        self.perm_loaded = True

    def __call__(self, input_patches):
        """
        The interface `__call__` is used to transform the input data. It should contain
        the actual implementation of data transform.

        Args:
            input_patches (List[torch.tensor]): list of torch tensors
        """
        if not self.perm_loaded:
            self._load_perms()
        perm_index = np.random.randint(self.perms.shape[0])
        shuffled_patches = [
            torch.FloatTensor(input_patches[i]) for i in self.perms[perm_index]
        ]
        # num_towers x C x H x W
        input_data = torch.stack(shuffled_patches)
        out_label = torch.Tensor([perm_index]).long()
        return input_data, out_label

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ShuffleImgPatches":
        """
        Instantiates ShuffleImgPatches from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ShuffleImgPatches instance.
        """
        assert "perm_file" in config, "Please specify the perm_file"
        return cls(perm_file=config["perm_file"])
