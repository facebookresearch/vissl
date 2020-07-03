# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Data and labels file for various datasets.
"""
import os

import numpy as np
from vissl.utils.slurm import get_slurm_dir


def get_local_path(input_file, dest_dir):
    out = ""
    if os.path.isfile(input_file):
        out = os.path.join(dest_dir, os.path.basename(input_file))
    elif os.path.isdir(input_file):
        data_name = input_file.strip("/").split("/")[-1]
        if "SLURM_JOBID" in os.environ:
            dest_dir = get_slurm_dir(dest_dir)
        dest_dir = os.path.join(dest_dir, data_name)
        complete_flag = os.path.join(dest_dir, "copy_complete")
        if os.path.isfile(complete_flag):
            out = dest_dir
    if os.path.exists(out):
        return out
    else:
        return input_file


def get_local_output_filepaths(input_files, dest_dir):
    # if we have copied the files to local disk as specified in the config, we
    # return those local paths. Otherwise return the original paths.
    output_files = []
    for item in input_files:
        if isinstance(item, list):
            out = get_local_output_filepaths(item, dest_dir)
        else:
            out = get_local_path(item, dest_dir)
        output_files.append(out)
    return output_files


def check_data_exists(data_files):
    if isinstance(data_files, list):
        return np.all([os.path.exists(item) for item in data_files])
    else:
        return os.path.exists(data_files)


def get_data_files(split, dataset_config):
    data_files = dataset_config[split].DATA_PATHS
    data_sources = dataset_config[split].DATA_SOURCES
    label_files, output_data_files = [], []
    # if there are synthetic data sources, we set the filepaths as none
    for idx in range(len(data_sources)):
        if data_sources[idx] == "synthetic":
            output_data_files.append("")
        else:
            output_data_files.append(data_files[idx])

    assert len(data_sources) == len(
        output_data_files
    ), "Mismatch between length of data_sources and data paths provided"
    if check_data_exists(dataset_config[split].LABEL_PATHS):
        label_files = dataset_config[split].LABEL_PATHS
    output = [output_data_files, label_files]
    if dataset_config[split].COPY_TO_LOCAL_DISK:
        dest_dir = dataset_config[split]["COPY_DESTINATION_DIR"]
        local_data_files = get_local_output_filepaths(data_files, dest_dir)
        local_label_files = get_local_output_filepaths(label_files, dest_dir)
        output = [local_data_files, local_label_files]
    return output
