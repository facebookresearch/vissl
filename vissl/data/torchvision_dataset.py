from typing import Tuple, List, Optional, Callable, Any

from PIL import Image
from fvcore.common.file_io import PathManager
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, STL10, MNIST

from vissl.utils.hydra_config import AttrDict


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
        is_train_split = self.split == "train"
        if self.dataset_name == "CIFAR10":
            return CIFAR10(self.path, train=is_train_split)
        elif self.dataset_name == "CIFAR100":
            return CIFAR100(self.path, train=is_train_split)
        elif self.dataset_name == "STL10":
            stl_split = "train" if is_train_split else "test"
            return STL10(self.path, split=stl_split)
        elif self.dataset_name == "MNIST":
            return MNISTAdapter(root=self.path, train=is_train_split)
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

    def get_labels(self) -> List[int]:
        """
        Return the labels for each sample
        """
        return [self.dataset[i][1] for i in range(self.num_samples())]


class MNISTAdapter(MNIST):
    """
    Wrapper around MNIST to convert images to RGB and change the size of the images
    to 32x32, to match the receptive field of a standard ResNet50.
    """

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        original_img, target = self.data[index], int(self.targets[index])
        original_img = Image.fromarray(original_img.numpy(), mode='L')
        img = Image.new(mode="RGB", size=(32, 32))
        img.paste(original_img, box=(2, 2, 30, 30))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
