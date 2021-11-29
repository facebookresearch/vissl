# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from iopath.common.file_io import g_pathmgr
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, STL10, SVHN
from vissl.config import AttrDict


class TorchvisionDatasetName:
    """
    Names of the Torchvision datasets currently supported in VISSL.
    """

    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    STL10 = "STL10"
    MNIST = "MNIST"
    SVHN = "SVHN"


class TorchvisionDataset(Dataset):
    """
    Adapter dataset to available datasets in Torchvision.
    The selected dataset is based on the name provided as argument.
    This name must match the names of the dataset in torchvision.

    Args:
        cfg (AttrDict): configuration defined by user
        data_source (string): data source ("torchvision_dataset") [not used]
        path (string): path to the dataset
        split (string): specify split for the dataset (either "train" or "val").
        dataset_name (string): name of dataset (should be one of TorchvisionDatasetName).
    """

    def __init__(
        self, cfg: AttrDict, data_source: str, path: str, split: str, dataset_name: str
    ):
        super().__init__()
        assert g_pathmgr.isdir(path), f"Directory {path} does not exist"
        self.dataset_name = dataset_name
        self.path = path
        self.split = split.lower()
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        is_train_split = self.split == "train"
        if self.dataset_name == TorchvisionDatasetName.CIFAR10:
            return CIFAR10(self.path, train=is_train_split)
        elif self.dataset_name == TorchvisionDatasetName.CIFAR100:
            return CIFAR100(self.path, train=is_train_split)
        elif self.dataset_name == TorchvisionDatasetName.STL10:
            stl_split = "train" if is_train_split else "test"
            return STL10(self.path, split=stl_split)
        elif self.dataset_name == TorchvisionDatasetName.MNIST:
            return MNIST(root=self.path, train=is_train_split)
        elif self.dataset_name == TorchvisionDatasetName.SVHN:
            stl_split = "train" if is_train_split else "test"
            return SVHN(root=self.path, split=stl_split)
        else:
            raise ValueError(f"Unsupported dataset {self.dataset_name}")

    def num_samples(self) -> int:
        """
        Size of the dataset
        """
        return len(self.dataset)

    def __len__(self) -> int:
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx: int) -> Tuple[Image.Image, bool]:
        """
        Return the image at index 'idx' and whether the load was successful
        """
        image = self.dataset[idx][0]
        is_success = True
        return image, is_success

    def get_labels(self) -> List[int]:
        """
        Return the labels for each sample
        """
        return [self.dataset[i][1] for i in range(self.num_samples())]
