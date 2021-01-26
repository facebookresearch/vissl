# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torch.utils.data import Dataset
from vissl.data.data_helper import get_mean_image


class SyntheticImageDataset(Dataset):
    """
    Synthetic dataset class. Mean image is returned always. This dataset
    is used/recommended to use for testing purposes only.

    Args:
        path (string): can be "" [not used]
        split (string): specify split for the dataset.
            Usually train/val/test. Used to read images if
            reading from a folder `path' and retrieve settings for that split
            from the config path [not used]
        dataset_name (string): name of dataset. For information only. [not used]
        data_source (string, Optional): data source ("synthetic") [not used]
    """

    def __init__(self, cfg, path, split, dataset_name, data_source="synthetic"):
        super(SyntheticImageDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.data_source = data_source
        self._num_samples = 50000
        # by default, pretend dataset size is 500 images. OR user specified limit
        if cfg.DATA[split].DATA_LIMIT > 0:
            self._num_samples = cfg.DATA[split].DATA_LIMIT

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def __len__(self):
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx):
        """
        Simply return the mean dummy image of the specified size and mark
        it as a success.
        """
        img = get_mean_image(self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE)
        is_success = True
        return img, is_success
