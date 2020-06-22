#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from vissl.dataset.data_helper import QueueDataset, get_mean_image


class DiskImageDataset(QueueDataset):
    """
    Base Dataset class for loading images from Disk.
    Can load a predefined list of images or all images inside
    a folder.

    Args
    path (string): can be either of the following
        1. A .npy file containing a list of filepaths.
        2. A folder such that folder/split contains images
    split (string): specify split for the dataset.
        Usually train/val/test. Used to read images if
        reading from a folder `path' and retrieve settings for that split
        from the config path
    dataset_name (string): name of dataset. For information only.
    data_source (string, Optional): data source ("disk") [Not used]
    """

    def __init__(self, cfg, path, split, dataset_name, data_source="disk"):
        # TODO: support a "ROOT_DIR" so that npy file contains relative paths to
        # ROOT_DIR
        # TODO: support sending back labels from ImageFolder
        super(DiskImageDataset, self).__init__(
            queue_size=cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        )
        assert os.path.exists(path), "Disk data path does NOT exist!"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        self.image_dataset = []
        # inferred parameter
        self._dataset_type = None
        self.is_initialized = False
        self._load_data(path)
        self._num_samples = len(self.image_dataset)
        # set dataset to null so that workers dont need to pickle this
        self.image_dataset = []
        # whether to use QueueDataset class to handle invalid images or not
        self.enable_queue_dataset = cfg["DATA"][self.split]["ENABLE_QUEUE_DATASET"]

    def _infer_dataset_type(self, path):
        if path.endswith("npy"):
            self._dataset_type = "file_list"
        elif os.path.isdir(path):
            self._dataset_type = "image_folder"
        else:
            raise "Unknown data type"

    def _load_data(self, path):
        self._infer_dataset_type(path)
        if self._dataset_type == "file_list":
            if self.cfg["DATA"][self.split].MMAP_MODE:
                self.image_dataset = np.load(path, mmap_mode="r")
            else:
                self.image_dataset = np.load(path)
        elif self._dataset_type == "image_folder":
            self.image_dataset = ImageFolder(path)

        if self.cfg["DATA"][self.split]["DATA_LIMIT"] > 0:
            limit = self.cfg["DATA"][self.split]["DATA_LIMIT"]
            if self._dataset_type == "file_list":
                self.image_dataset = self.image_dataset[:limit]
            elif self._dataset_type == "image_folder":
                self.image_dataset.samples = self.image_dataset.samples[:limit]

    def num_samples(self):
        return self._num_samples

    def get_image_paths(self):
        self._load_data(self._path)
        return self.image_dataset

    def __len__(self):
        return self.num_samples()

    def __getitem__(self, idx):
        if not self.is_initialized:
            self._load_data(self._path)
            self.is_initialized = True
        if not self.queue_init and self.enable_queue_dataset:
            self._init_queues()
        is_success = True
        image_path = self.image_dataset[idx]
        try:
            if self._dataset_type == "image_folder":
                img = self.image_dataset[idx][0]
            else:
                img = Image.open(image_path).convert("RGB")
            if is_success and self.enable_queue_dataset:
                self.on_sucess(img)
        except Exception as e:
            if self.cfg.VERBOSE:
                logging.warn(
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
