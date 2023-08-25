# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Data and labels file for various datasets.
"""

import json
import logging
import os
from typing import List

import numpy as np
from iopath.common.file_io import g_pathmgr
from vissl.data.datasets import get_coco_imgs_labels_info, get_voc_images_labels_info
from vissl.utils.misc import get_json_data_catalog_file
from vissl.utils.slurm import get_slurm_dir


class VisslDatasetCatalog:
    """
    A catalog that stores information about the datasets and how to obtain them.
    It contains a mapping from strings (which are names that identify a dataset,
    e.g. "imagenet1k") to a `dict` which contains:
        1) mapping of various data splits (train, test, val) to the data source
           (path on the disk whether a folder path or a filelist)
        2) source of the data (disk_filelist | disk_folder | disk_roi_annotations)
    The purpose of having this catalog is to make it easy to choose different datasets,
    by just using the strings in the config.
    """

    __REGISTERED_DATASETS = {}

    @staticmethod
    def register_json(json_catalog_path):
        """
        Args:
            filepath: a .json filepath that contains the data to be registered
        """
        with g_pathmgr.open(json_catalog_path) as fopen:
            data_catalog = json.load(fopen)
        for key, value in data_catalog.items():
            VisslDatasetCatalog.register_data(key, value)

    @staticmethod
    def register_dict(dict_catalog):
        """
        Args:
            dict: a dict with a bunch of datasets to be registered
        """
        for key, value in dict_catalog.items():
            VisslDatasetCatalog.register_data(key, value)

    @staticmethod
    def register_data(name, data_dict):
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "imagenet1k_folder".
            func (callable): a callable which takes no arguments and returns a list of dicts.
                It must return the same results if called multiple times.
        """
        assert isinstance(
            data_dict, dict
        ), "You must register a dictionary with VisslDatasetCatalog.register_dict"
        assert (
            name not in VisslDatasetCatalog.__REGISTERED_DATASETS
        ), "Dataset '{}' is already registered!".format(name)
        VisslDatasetCatalog.__REGISTERED_DATASETS[name] = data_dict

    @staticmethod
    def get(name):
        """
        Get the registered dict and return it.

        Args:
            name (str): the name that identifies a dataset, e.g. "imagenet1k".
        Returns:
            dict: dataset information (paths, source)
        """
        try:
            info = VisslDatasetCatalog.__REGISTERED_DATASETS[name]
        except KeyError:
            raise KeyError(
                "Dataset '{}' is not registered! Available datasets are: {}".format(
                    name, ", ".join(VisslDatasetCatalog.__REGISTERED_DATASETS.keys())
                )
            )
        return info

    @staticmethod
    def list() -> List[str]:
        """
        List all registered datasets.

        Returns:
            list[str]
        """
        return list(VisslDatasetCatalog.__REGISTERED_DATASETS.keys())

    @staticmethod
    def clear():
        """
        Remove all registered dataset.
        """
        VisslDatasetCatalog.__REGISTERED_DATASETS.clear()

    @staticmethod
    def remove(name):
        """
        Remove the dataset registered by ``name``.
        """
        VisslDatasetCatalog.__REGISTERED_DATASETS.pop(name)

    @staticmethod
    def has_data(name):
        """
        Check whether the data with ``name`` exists.
        """
        data_found = name in VisslDatasetCatalog.__REGISTERED_DATASETS
        return data_found


def get_local_path(input_file, dest_dir):
    """
    If user specified copying data to a local directory,
    get the local path where the data files were copied.

    - If input_file is just a file, we return the dest_dir/filename
    - If the intput_file is a directory, then we check if the
      environemt is SLURM and use slurm_dir or otherwise dest_dir
      to look up copy_complete file is available.
      If available, we return the directory.
    - If both above fail, we return the input_file as is.
    """
    out = ""
    if g_pathmgr.isfile(input_file):
        out = os.path.join(dest_dir, os.path.basename(input_file))
    elif g_pathmgr.isdir(input_file):
        data_name = input_file.strip("/").split("/")[-1]
        if "SLURM_JOBID" in os.environ:
            dest_dir = get_slurm_dir(dest_dir)
        dest_dir = os.path.join(dest_dir, data_name)
        complete_flag = os.path.join(dest_dir, "copy_complete")
        if g_pathmgr.isfile(complete_flag):
            out = dest_dir
    if g_pathmgr.exists(out):
        return out
    else:
        return input_file


def get_local_output_filepaths(input_files, dest_dir):
    """
    If we have copied the files to local disk as specified in the config, we
    return those local paths. Otherwise return the original paths.
    """
    output_files = []
    for item in input_files:
        if isinstance(item, list):
            out = get_local_output_filepaths(item, dest_dir)
        else:
            out = get_local_path(item, dest_dir)
        output_files.append(out)
    return output_files


def check_data_exists(data_files):
    """
    Check that the input data files exist. If the data_files is a list,
    we iteratively check for each file in the list.
    """
    if isinstance(data_files, list):
        return np.all([g_pathmgr.exists(item) for item in data_files])
    else:
        return g_pathmgr.exists(data_files)


def register_pascal_voc():
    """
    Register PASCAL VOC 2007 and 2012 datasets to the data catalog.
    We first look up for these datasets paths in the dataset catalog,
    if the paths exist, we register, otherwise we remove the voc_data
    from the catalog registry.
    """
    voc_datasets = ["voc2007_folder", "voc2012_folder"]
    for voc_data in voc_datasets:
        data_info = VisslDatasetCatalog.get(voc_data)
        data_folder = data_info["train"][0]
        if g_pathmgr.exists(data_folder):
            train_data_info = get_voc_images_labels_info("train", data_folder)
            test_data_info = get_voc_images_labels_info("val", data_folder)
            data_info["train"] = train_data_info
            data_info["val"] = test_data_info
            VisslDatasetCatalog.remove(voc_data)
            VisslDatasetCatalog.register_data(voc_data, data_info)
        else:
            VisslDatasetCatalog.remove(voc_data)


def register_coco():
    """
    Register COCO 2004 datasets to the data catalog.
    We first look up for these datasets paths in the dataset catalog,
    if the paths exist, we register, otherwise we remove the
    coco2014_folder from the catalog registry.
    """
    data_info = VisslDatasetCatalog.get("coco2014_folder")
    data_folder = data_info["train"][0]
    if g_pathmgr.exists(data_folder):
        train_data_info = get_coco_imgs_labels_info("train", data_folder)
        test_data_info = get_coco_imgs_labels_info("val", data_folder)
        data_info["train"] = train_data_info
        data_info["val"] = test_data_info
        VisslDatasetCatalog.remove("coco2014_folder")
        VisslDatasetCatalog.register_data("coco2014_folder", data_info)
    else:
        VisslDatasetCatalog.remove("coco2014_folder")


def register_datasets(json_catalog_path):
    """
    If the json dataset_catalog file is found, we register
    the datasets specified in the catalog with VISSL.
    If the catalog also specified VOC or coco datasets, we resister them

    Args:
        json_catalog_path (str): the path to the json dataset catalog
    """
    if g_pathmgr.exists(json_catalog_path):
        logging.info(f"Registering datasets: {json_catalog_path}")
        VisslDatasetCatalog.clear()
        VisslDatasetCatalog.register_json(json_catalog_path)
    if VisslDatasetCatalog.has_data("voc2007_folder") or VisslDatasetCatalog.has_data(
        "voc2012_folder"
    ):
        register_pascal_voc()
    if VisslDatasetCatalog.has_data("coco2014_folder"):
        register_coco()


def get_data_files(split, dataset_config):
    """
    Get the path to the dataset (images and labels).
        1. If the user has explicitly specified the data_sources, we simply
           use those and don't do lookup in the datasets registered with VISSL
           from the dataset catalog.
        2. If the user hasn't specified the path, look for the dataset in
           the datasets catalog registered with VISSL. For a given list of datasets
           and a given partition (train/test), we first verify that we have the
           dataset and the correct source as specified by the user.
           Then for each dataset in the list, we get the data path (make sure it
           exists, sources match). For the label file, the file is optional.

    Once we have the dataset original paths, we replace the path with the local paths
    if the data was copied to local disk.
    """
    assert len(dataset_config[split].DATASET_NAMES) == len(
        dataset_config[split].DATA_SOURCES
    ), "len(data_sources) != len(dataset_names)"
    if len(dataset_config[split].DATA_PATHS) > 0:
        assert len(dataset_config[split].DATA_SOURCES) == len(
            dataset_config[split].DATA_PATHS
        ), "len(data_sources) != len(data_paths)"
    data_files, label_files = [], []
    data_names = dataset_config[split].DATASET_NAMES
    data_sources = dataset_config[split].DATA_SOURCES
    data_split = "train" if split == "TRAIN" else "val"
    for idx in range(len(data_sources)):
        # if there are synthetic data sources, we set the filepaths as none
        if data_sources[idx] == "synthetic":
            data_files.append("")
            continue
        # if user has specified the data path explicitly, we use it
        elif len(dataset_config[split].DATA_PATHS) > 0:
            data_files.append(dataset_config[split].DATA_PATHS[idx])
        # otherwise retrieve from the cataloag based on the dataset name
        else:
            data_info = VisslDatasetCatalog.get(data_names[idx])
            assert (
                len(data_info[data_split]) > 0
            ), f"data paths list for split: { data_split } is empty"
            check_data_exists(
                data_info[data_split][0]
            ), f"Some data files dont exist: {data_info[data_split][0]}"
            data_files.append(data_info[data_split][0])
        # labels are optional and hence we append if we find them
        if len(dataset_config[split].LABEL_PATHS) > 0:
            if check_data_exists(dataset_config[split].LABEL_PATHS[idx]):
                label_files.append(dataset_config[split].LABEL_PATHS[idx])
        else:
            label_data_info = VisslDatasetCatalog.get(data_names[idx])
            if check_data_exists(label_data_info[data_split][1]):
                label_files.append(label_data_info[data_split][1])

    output = [data_files, label_files]
    if dataset_config[split].COPY_TO_LOCAL_DISK:
        dest_dir = dataset_config[split]["COPY_DESTINATION_DIR"]
        local_data_files = get_local_output_filepaths(data_files, dest_dir)
        local_label_files = get_local_output_filepaths(label_files, dest_dir)
        output = [local_data_files, local_label_files]
    return output


# get the path to dataset_catalog.json file
json_catalog_file = get_json_data_catalog_file()
# register the datasets specified in the catalog with VISSL
register_datasets(json_catalog_file)
