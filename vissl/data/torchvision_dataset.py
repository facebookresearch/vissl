from typing import Tuple

from PIL import Image
from fvcore.common.file_io import PathManager
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from vissl.utils.hydra_config import AttrDict


class PytorchImageDataset(Dataset):
    """
    Adapter dataset to available datasets in Torchvision.
    The selected dataset is based on the name provided as argument.
    This name must match the names of the dataset in torchvision.

    Args:
        cfg (AttrDict): configuration defined by user
        data_source (string): data source ("pytorch_dataset") [not used]
        path (string): path to the dataset
        split (string): specify split for the dataset (either "train" or "val").
        dataset_name (string): name of dataset.
    """

    def __init__(self,
                 cfg: AttrDict,
                 data_source: str,
                 path: str,
                 split: str,
                 dataset_name: str):
        super().__init__()
        assert PathManager.isdir(path), f"Directory {path} does not exist"
        self.dataset_name = dataset_name
        self.path = path
        self.split = split.lower()
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        if self.dataset_name == "CIFAR10":
            return CIFAR10(self.path, train=self.split == "train")
        elif self.dataset_name == "CIFAR100":
            return CIFAR100(self.path, train=self.split == "train")
        else:
            raise ValueError(f"Unsupported dataset {self.dataset_name: str}")

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
        Note: we do delayed loading of data to reduce the memory size
              due to pickling of dataset across dataloader workers.
        """
        image = self.dataset[idx][0]
        is_success = True
        return image, is_success

    def get_image_and_label(self, idx: int) -> Tuple[Image.Image, int]:
        sample = self.dataset[idx]
        return sample[0], sample[1]
