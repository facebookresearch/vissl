# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import numpy as np
from classy_vision.generic.distributed_util import get_world_size
from fvcore.common.file_io import PathManager
from torch.utils.data import Dataset
from vissl.data import dataset_catalog
from vissl.data.ssl_transforms import get_transform
from vissl.utils.env import get_machine_local_and_dist_rank


class GenericSSLDataset(Dataset):
    """
    Base Self Supervised Learning Dataset Class.
    TODO:: Documentation
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
        for idx in range(len(self.data_sources)):
            assert self.data_sources[idx] in dataset_source_map, (
                f"Unknown data source: {self.data_sources[idx]}, supported: "
                f"{list(dataset_source_map.keys())}"
            )

    def _get_data_files(self, split):
        local_rank, _ = get_machine_local_and_dist_rank()
        self.data_paths, self.label_paths = dataset_catalog.get_data_files(
            split, dataset_config=self.cfg["DATA"]
        )

        logging.info(f"Rank: {local_rank} Data files:\n{self.data_paths}")
        logging.info(f"Rank: {local_rank} Label files:\n{self.label_paths}")

    def _load_labels(self):
        local_rank, _ = get_machine_local_and_dist_rank()
        for idx, label_source in enumerate(self.label_sources):
            if label_source == "disk_filelist":
                path = self.label_paths[idx]
                # Labels are stored in a file
                assert PathManager.isfile(path), f"Path to labels {path} is not a file"

                assert path.endswith("npy"), "Please specify a numpy file for labels"
                if self.cfg["DATA"][self.split].MMAP_MODE:
                    # Memory map the labels if the file is too large.
                    # This is useful to save RAM.
                    labels = np.load(path, mmap_mode="r")
                else:
                    labels = np.load(path)
                # if the labels are int32, we convert them to int64 since pytorch
                # needs a long (int64) type for labels to index. See
                # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/5  # NOQA
                if labels.dtype == np.int32:
                    labels = labels.astype(np.int64)
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
                item["label"].append(source[idx])
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
        return len(self.data_objs[0])

    def get_image_paths(self):
        image_paths = []
        for source in self.data_objs:
            image_paths.append(source.get_image_paths())
        return image_paths

    def get_available_splits(self, dataset_config):
        return [key for key in dataset_config if key.lower() in ["train", "test"]]

    def num_samples(self, source_idx=0):
        return len(self.data_objs[source_idx])

    def get_batchsize_per_replica(self):
        # this searches for batchsize_per_replica in self and then in self.dataset
        return getattr(self, "batchsize_per_replica", 1)

    def get_global_batchsize(self):
        return self.get_batchsize_per_replica() * get_world_size()
