# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from iopath.common.file_io import g_pathmgr
from PIL import Image
from torchvision.datasets import ImageFolder
from vissl.data.data_helper import get_mean_image, QueueDataset
from vissl.utils.io import load_file


class DiskImageDataset(QueueDataset):
    """
    Base Dataset class for loading images from Disk.
    Can load a predefined list of images or all images inside
    a folder. And also a json file containing the image path
    and the RoI annotation for the image.

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
            3. A .json file containing list of dictionary where
               each dictionary has "img_path" and "bbox" entry.
               In this case `data_source = "disk_roi_annotations"`
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
            "disk_roi_annotations",
        ], "data_source must be either disk_filelist or disk_folder"
        if data_source == "disk_filelist":
            assert g_pathmgr.isfile(path), f"File {path} does not exist"
        elif data_source == "disk_folder":
            assert g_pathmgr.isdir(path), f"Directory {path} does not exist"
        elif data_source == "disk_roi_annotations":
            assert g_pathmgr.isfile(path), f"File {path} does not exist"
            assert path.endswith("json"), "Annotations must be in json format"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        self.image_dataset = []
        self.image_roi_bbox = []
        self.is_initialized = False
        self._load_data(path)
        self._num_samples = len(self.image_dataset)
        self._remove_prefix = cfg["DATA"][self.split]["REMOVE_IMG_PATH_PREFIX"]
        self._new_prefix = cfg["DATA"][self.split]["NEW_IMG_PATH_PREFIX"]
        if self.data_source in ["disk_filelist", "disk_roi_annotations"]:
            # Set dataset to null so that workers dont need to pickle this file.
            # This saves memory when disk_filelist is large, especially when memory mapping.
            self.image_dataset = []
            self.image_roi_bbox = []
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
        elif self.data_source == "disk_roi_annotations":
            # we load the annotations and then parse the image paths and the image roi
            self.image_dataset, self.image_roi_bbox = [], []
            json_annotations = load_file(path)
            self.image_dataset = [item["path"] for item in json_annotations]
            self.image_roi_bbox = [item["bbox"] for item in json_annotations]

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
        if self.data_source == "disk_folder":
            assert isinstance(self.image_dataset, ImageFolder)
            return [sample[0] for sample in self.image_dataset.samples]
        else:
            return self.image_dataset

    @staticmethod
    def _replace_img_path_prefix(img_path: str, replace_prefix: str, new_prefix: str):
        if img_path.startswith(replace_prefix):
            return img_path.replace(replace_prefix, new_prefix)
        return img_path

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
                image_path = self._replace_img_path_prefix(
                    image_path,
                    replace_prefix=self._remove_prefix,
                    new_prefix=self._new_prefix,
                )
                with g_pathmgr.open(image_path, "rb") as fopen:
                    img = Image.open(fopen).convert("RGB")
            elif self.data_source == "disk_folder":
                img = self.image_dataset[idx][0]
            elif self.data_source == "disk_roi_annotations":
                with g_pathmgr.open(image_path, "rb") as fopen:
                    bbox = [float(item) for item in self.image_roi_bbox[idx]]
                    img = Image.open(fopen).crop(bbox).convert("RGB")
                # TODO: move the below to a dedicated transform
                # applicable to openimages dataset only
                width, height = img.size
                bbox_size = min(img.size)
                ratio = max(img.size) / min(img.size)
                if ratio >= 1.2:
                    if width < height:  # bigger height
                        bbox = (0, 0, bbox_size, bbox_size)
                    else:  # bigger width.
                        bbox = (
                            int((width - bbox_size) / 2),
                            0,
                            int((width - bbox_size) / 2) + bbox_size,
                            height,
                        )
                    img = img.crop(bbox)
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
