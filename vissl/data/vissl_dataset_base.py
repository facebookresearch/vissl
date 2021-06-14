# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Callable, Dict, Set

import torch
from vissl.config import AttrDict


class VisslDatasetBase(abc.ABC):
    """
    Vissl Dataset Base class. Used to create common interface among
    base datasets. e.g. GenericSSLDataset
    """

    def __init__(
        self,
        cfg: AttrDict,
        split: str,
        dataset_source_map: Dict[str, Callable],
        data_sources_with_subset: Set[str],
        device: torch.device,
    ):
        self.cfg = cfg
        self.split = split
        self.dataset_source_map = dataset_source_map
        self.data_sources_with_subset = data_sources_with_subset
        self.device = device

    @abc.abstractmethod
    def num_samples(self):
        """
        Number of samples.
        """
        ...

    @abc.abstractmethod
    def get_global_batchsize(self):
        """
        batch_size_per_replica * world_size
        """
        ...
