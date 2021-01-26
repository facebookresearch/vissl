# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

from fvcore.common.file_io import PathManager
from PIL import Image
from torchvision.datasets import ImageFolder
from vissl.data.data_helper import QueueDataset, get_mean_image
from vissl.utils.io import load_file


class DiskImageDataset(QueueDataset):
    """
    Base Dataset class for loading images from Disk.
    Can load a predefined list of images or all images inside
    a folder.

    Inherits from QueueDataset class in VISSL to provide better
    handling of the invalid images by replacing them with the
    valid and seen images.

    Args:
        cfg (AttrDict): configuration defined by user
        data_source (string): data source either of "disk_filelist" or "disk_folder"
        path (string): can be either of the following
            1. A .npy file containing a list of filepaths.
               In this case `data_source = "disk_filelist"`
            2. A folder such that folder/split contains images.
               In this case `data_source = "disk_folder"`
        split (string): specify split for the dataset.
                        Usually train/val/test.
                        Used to read images if reading from a folder `path` and retrieve
                        settings for that split from the config path.
        dataset_name (string): name of dataset. For information only.

    NOTE: This dataset class only returns images (not labels or other metdata).
    To load labels you must specify them in `LABEL_SOURCES` (See `ssl_dataset.py`).
    LABEL_SOURCES follows a similar convention as the dataset and can either be a filelist
    or a torchvision ImageFolder compatible folder -
    1. Store labels in a numpy file
    2. Store images in a nested directory structure so that torchvision ImageFolder
       dataset can infer the labels.
    """

    def __init__(self, cfg, data_source, path, split, dataset_name):
        super(DiskImageDataset, self).__init__(
            queue_size=cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        )
        assert data_source in [
            "disk_filelist",
            "disk_folder",
        ], "data_source must be either disk_filelist or disk_folder"
        if data_source == "disk_filelist":
            assert PathManager.isfile(path), f"File {path} does not exist"
        elif data_source == "disk_folder":
            assert PathManager.isdir(path), f"Directory {path} does not exist"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        self.image_dataset = []
        self.is_initialized = False
        self._load_data(path)
        self._num_samples = len(self.image_dataset)
        if self.data_source == "disk_filelist":
            # Set dataset to null so that workers dont need to pickle this file.
            # This saves memory when disk_filelist is large, especially when memory mapping.
            self.image_dataset = []
        # whether to use QueueDataset class to handle invalid images or not
        self.enable_queue_dataset = cfg["DATA"][self.split]["ENABLE_QUEUE_DATASET"]

    def _load_data(self, path):
        if self.data_source == "disk_filelist":
            if self.cfg["DATA"][self.split].MMAP_MODE:
                self.image_dataset = load_file(path, mmap_mode="r")
            else:
                self.image_dataset = load_file(path)
        elif self.data_source == "disk_folder":
            self.image_dataset = ImageFolder(path)
            logging.info(f"Loaded {len(self.image_dataset)} samples from folder {path}")

            # mark as initialized.
            # Creating ImageFolder dataset can be expensive because of repeated os.listdir calls
            # Avoid creating it over and over again.
            self.is_initialized = True

        if self.cfg["DATA"][self.split]["DATA_LIMIT"] > 0:
            limit = self.cfg["DATA"][self.split]["DATA_LIMIT"]
            if self.data_source == "disk_filelist":
                self.image_dataset = self.image_dataset[:limit]
            elif self.data_source == "disk_folder":
                self.image_dataset.samples = self.image_dataset.samples[:limit]

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def get_image_paths(self):
        """
        Get paths of all images in the datasets. See load_data()
        """
        self._load_data(self._path)
        return self.image_dataset

    def __len__(self):
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx):
        """
        - We do delayed loading of data to reduce the memory size due to pickling of
          dataset across dataloader workers.
        - Loads the data if not already loaded.
        - Sets and initializes the queue if not already initialized
        - Depending on the data source (folder or filelist), get the image.
          If using the QueueDataset and image is valid, save the image in queue if
          not full. Otherwise return a valid seen image from the queue if queue is
          not empty.
        """
        if not self.is_initialized:
            self._load_data(self._path)
            self.is_initialized = True
        if not self.queue_init and self.enable_queue_dataset:
            self._init_queues()
        is_success = True
        image_path = self.image_dataset[idx]
        try:
            if self.data_source == "disk_filelist":
                with PathManager.open(image_path, "rb") as fopen:
                    img = Image.open(fopen).convert("RGB")
            elif self.data_source == "disk_folder":
                img = self.image_dataset[idx][0]
            if is_success and self.enable_queue_dataset:
                self.on_sucess(img)
        except Exception as e:
            logging.warning(
                f"Couldn't load: {self.image_dataset[idx]}. Exception: \n{e}"
            )
            is_success = False
            # if we have queue dataset class enabled, we try to use it to get
            # the seen valid images
            if self.enable_queue_dataset:
                img, is_success = self.on_failure()
                if img is None:
                    img = get_mean_image(
                        self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE
                    )
            else:
                img = get_mean_image(self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE)
        return img, is_success
