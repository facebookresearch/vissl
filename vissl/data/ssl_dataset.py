# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import numpy as np
from classy_vision.generic.distributed_util import get_world_size
from fvcore.common.file_io import PathManager
from torch.utils.data import Dataset
from vissl.data import dataset_catalog
from vissl.data.ssl_transforms import get_transform
from vissl.utils.env import get_machine_local_and_dist_rank


def _convert_lbl_to_long(lbl):
    """
    if the labels are int32, we convert them to int64 since pytorch
    needs a long (int64) type for labels to index. See
    https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/5  # NOQA
    """
    out_lbl = lbl
    if isinstance(lbl, np.ndarray) and (lbl.dtype == np.int32):
        out_lbl = lbl.astype(np.int64)
    elif isinstance(lbl, list):
        out_lbl = [_convert_lbl_to_long(item) for item in lbl]
    elif isinstance(lbl, np.int32):
        out_lbl = out_lbl.astype(np.int64)
    return out_lbl


class GenericSSLDataset(Dataset):
    """
    Base Self Supervised Learning Dataset Class.

    The GenericSSLDataset class is defined to support reading data
    from multiple data sources. For example: data = [dataset1, dataset2]
    and the minibatches generated will have the corresponding data
    from each dataset.

    For this reason, we also support labels from multiple sources. For example
    targets = [dataset1 targets, dataset2 targets].

    In order to support multiple data sources, the dataset configuration
    always has list inputs.
        - DATA_SOURCES, LABEL_SOURCES, DATASET_NAMES, DATA_PATHS, LABEL_PATHS

    For several data sources, we also support specifying on what dataset the
    transforms should be applied. By default, apply the transforms
    on data from all datasets.

    Args:
        cfg (AttrDict): configuration defined by user
        split (str): the dataset split for which we are constructing the Dataset object
        dataset_source_map (Dict[str, Callable]): The dictionary that maps
                    what data sources are supported and what object to use to read
                    data from those sources. For example:
                    DATASET_SOURCE_MAP = {
                        "disk_filelist": DiskImageDataset,
                        "disk_folder": DiskImageDataset,
                        "synthetic": SyntheticImageDataset,
                    }
    """

    def __init__(self, cfg, split, dataset_source_map):
        self.split = split
        self.cfg = cfg
        self.data_objs = []
        self.label_objs = []
        self.data_paths = []
        self.label_paths = []
        self.batchsize_per_replica = self.cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        self.data_sources = self.cfg["DATA"][split].DATA_SOURCES
        self.label_sources = self.cfg["DATA"][split].LABEL_SOURCES
        self.dataset_names = self.cfg["DATA"][split].DATASET_NAMES
        self.label_type = self.cfg["DATA"][split].LABEL_TYPE
        self.transform = get_transform(self.cfg["DATA"][split].TRANSFORMS)
        self._labels_init = False
        self._verify_data_sources(split, dataset_source_map)
        self._get_data_files(split)

        if len(self.label_sources) > 0 and len(self.label_paths) > 0:
            assert len(self.label_sources) == len(self.label_paths), (
                f"len(label_sources) != len(label paths) "
                f"{len(self.label_sources)} vs. {len(self.label_paths)}"
            )

        for idx in range(len(self.data_sources)):
            datasource_cls = dataset_source_map[self.data_sources[idx]]
            self.data_objs.append(
                datasource_cls(
                    cfg=self.cfg,
                    path=self.data_paths[idx],
                    split=split,
                    dataset_name=self.dataset_names[idx],
                    data_source=self.data_sources[idx],
                )
            )

    def _verify_data_sources(self, split, dataset_source_map):
        """
        For each data source, verify that the specified data source
        is supported in VISSL. See DATASET_SOURCE_MAP for what sources
        are supported.
        """
        for idx in range(len(self.data_sources)):
            assert self.data_sources[idx] in dataset_source_map, (
                f"Unknown data source: {self.data_sources[idx]}, supported: "
                f"{list(dataset_source_map.keys())}"
            )

    def _get_data_files(self, split):
        """
        Get the given dataset split (train or test), get the path to the dataset
        (images and labels).
        1. If the user has explicitly specified the data_sources, we simply
           use those and don't do lookup in the datasets registered with VISSL
           from the dataset catalog.
        2. If the user hasn't specified the path, look for the dataset in
           the datasets catalog registered with VISSL. For a given list of datasets
           and a given partition (train/test), we first verify that we have the
           dataset and the correct source as specified by the user.
           Then for each dataset in the list, we get the data path (make sure it
           exists, sources match). For the label file, the file is optional.
        """
        local_rank, _ = get_machine_local_and_dist_rank()
        self.data_paths, self.label_paths = dataset_catalog.get_data_files(
            split, dataset_config=self.cfg["DATA"]
        )

        logging.info(
            f"Rank: {local_rank} split: {split} Data files:\n{self.data_paths}"
        )
        logging.info(
            f"Rank: {local_rank} split: {split} Label files:\n{self.label_paths}"
        )

    def load_single_label_file(self, path):
        """
        Load the single data file. We only support user specifying the numpy label
        files if user is specifying a data_filelist source of labels.

        To save memory, if the mmap_mode is set to True for loading, we try to load
        the images in mmap_mode. If it fails, we simply load the labels without mmap
        """
        assert PathManager.isfile(path), f"Path to labels {path} is not a file"
        assert path.endswith("npy"), "Please specify a numpy file for labels"
        if self.cfg["DATA"][self.split].MMAP_MODE:
            try:
                with PathManager.open(path, "rb") as fopen:
                    labels = np.load(fopen, allow_pickle=True, mmap_mode="r")
            except ValueError as e:
                logging.info(f"Could not mmap {path}: {e}. Trying without PathManager")
                labels = np.load(path, allow_pickle=True, mmap_mode="r")
                logging.info("Successfully loaded without PathManager")
            except Exception:
                logging.info("Could not mmap without PathManager. Trying without mmap")
                with PathManager.open(path, "rb") as fopen:
                    labels = np.load(fopen, allow_pickle=True)
        else:
            with PathManager.open(path, "rb") as fopen:
                labels = np.load(fopen, allow_pickle=True)
        return labels

    def _load_labels(self):
        """
        Load the labels if the dataset has labels. In self-supervised
        pre-training task, we don't use labels. However, we use labels for the
        evaluations of the self-supervised models on the downstream tasks.

        For labels, two label sources are supported: disk_filelist and disk_folder

        In case of disk_filelist, we iteratively read labels for each specified file.
        See load_single_label_file().
        In case of disk_folder, we use the ImageFolder object created during the
        data loading itself.
        """
        local_rank, _ = get_machine_local_and_dist_rank()
        for idx, label_source in enumerate(self.label_sources):
            if label_source == "disk_filelist":
                paths = self.label_paths[idx]
                # in case of filelist, we support multiple label files.
                # we rely on the user to have a proper collator to handle
                # the multiple labels
                logging.info(f"Loading labels: {paths}")
                if isinstance(paths, list):
                    labels = []
                    for path in paths:
                        path_labels = self.load_single_label_file(path)
                        labels.append(path_labels)
                else:
                    labels = self.load_single_label_file(paths)
            elif label_source == "disk_folder":
                # In this case we use the labels inferred from the directory structure
                # We enforce that the data source also be a disk folder in this case
                assert self.data_sources[idx] == self.label_sources[idx]
                if local_rank == 0:
                    logging.info(
                        f"Using {label_source} labels from {self.data_paths[idx]}"
                    )
                # Use the ImageFolder object created when loading images.
                # We do not create it again since it can be an expensive operation.
                labels = [x[1] for x in self.data_objs[idx].image_dataset.samples]
                labels = np.array(labels).astype(np.int64)
            else:
                raise ValueError(f"unknown label source: {label_source}")
            self.label_objs.append(labels)

    def __getitem__(self, idx):
        """
        Get the input sample for the minibatch for a specified data index.
        For each data object (if we are loading several datasets in a minibatch),
        we get the sample: consisting of {
            - image data,
            - label (if applicable) otherwise idx
            - data_valid: 0 or 1 indicating if the data is valid image
            - data_idx : index of the data in the dataset for book-keeping and debugging
        }

        Once the sample data is available, we apply the data transform on the sample.

        The final transformed sample is returned to be added into the minibatch.
        """

        if not self._labels_init and len(self.label_sources) > 0:
            self._load_labels()
            self._labels_init = True

        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        item = {"data": [], "data_valid": [], "data_idx": []}
        for source in self.data_objs:
            data, valid = source[idx]
            item["data"].append(data)
            item["data_idx"].append(idx)
            item["data_valid"].append(1 if valid else -1)

        if (len(self.label_objs) > 0) or self.label_type == "standard":
            item["label"] = []
            for source in self.label_objs:
                if isinstance(source, list):
                    lbl = [entry[idx] for entry in source]
                else:
                    lbl = _convert_lbl_to_long(source[idx])
                item["label"].append(lbl)
        elif self.label_type == "sample_index":
            item["label"] = []
            for _ in range(len(self.data_objs)):
                item["label"].append(idx)
        else:
            raise ValueError(f"Unknown label type: {self.label_type}")

        # apply the transforms on the image
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        """
        Size of the dataset. Assumption made there is only one
        data source
        """
        return len(self.data_objs[0])

    def get_image_paths(self):
        """
        Get the image paths for all the data sources.

        Return:
            image_paths (List[List[str]]): list containing image paths list for each
                                            data source.
        """
        image_paths = []
        for source in self.data_objs:
            image_paths.append(source.get_image_paths())
        return image_paths

    def get_available_splits(self, dataset_config):
        """
        Get the available splits in the dataset confir. Not specific to this split
        for which the SSLDataset is being constructed.

        NOTE: this is deprecated method.
        """
        return [key for key in dataset_config if key.lower() in ["train", "test"]]

    def num_samples(self, source_idx=0):
        """
        Size of the dataset. Assumption made there is only one
        data source
        """
        return len(self.data_objs[source_idx])

    def get_batchsize_per_replica(self):
        """
        Get the batch size per trainer
        """
        # this searches for batchsize_per_replica in self and then in self.dataset
        return getattr(self, "batchsize_per_replica", 1)

    def get_global_batchsize(self):
        """
        The global batch size across all the trainers
        """
        return self.get_batchsize_per_replica() * get_world_size()
