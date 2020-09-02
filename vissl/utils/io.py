# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import logging
import os
import pickle
import re
import time

import numpy as np
from fvcore.common.file_io import PathManager
from vissl.utils.slurm import get_slurm_dir


def save_file(data, filename):
    logging.info(f"Saving data to file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with PathManager.open(filename, "wb") as fopen:
            pickle.dump(data, fopen, pickle.HIGHEST_PROTOCOL)
    elif file_ext == ".npy":
        with PathManager.open(filename, "wb") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        with PathManager.open(filename, "a") as fopen:
            fopen.write(json.dumps(data, sort_keys=True) + "\n")
            fopen.flush()
    else:
        raise Exception(f"Saving {file_ext} is not supported yet")
    logging.info(f"Saved data to file: {filename}")


def load_file(filename):
    logging.info(f"Loading data from file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with PathManager.open(filename, "rb") as fopen:
            data = pickle.load(fopen, encoding="latin1")
    elif file_ext == ".npy":
        with PathManager.open(filename, "rb") as fopen:
            data = np.load(fopen, encoding="latin1")
    elif file_ext == ".json":
        with PathManager.open(filename, "r") as fopen:
            data = json.loads(fopen)
    else:
        raise Exception(f"Reading from {file_ext} is not supported yet")
    return data


def makedir(dir_path):
    is_success = False
    try:
        if not PathManager.exists(dir_path):
            PathManager.mkdirs(dir_path)
        is_success = True
    except BaseException:
        logging.info(f"Error creating directory: {dir_path}")
    return is_success


def is_url(input_url):
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url


def cleanup_dir(dir):
    if PathManager.exists(dir):
        logging.info(f"Deleting directory: {dir}")
        os.system(f"rm -rf {dir}")
    logging.info(f"Deleted contents of directory: {dir}")


def get_file_size(filename):
    size_in_mb = os.path.getsize(filename) / float(1024 ** 2)
    return size_in_mb


def copy_file(input_file, destination_dir):
    destination_dir = get_slurm_dir(destination_dir)
    if "SLURM_JOBID" in os.environ:
        destination_dir = get_slurm_dir(destination_dir)
    makedir(destination_dir)
    output_file = f"{destination_dir}/{os.path.basename(input_file)}"
    if PathManager.exists(output_file):
        logging.info(f"File already copied: {output_file}")
        return output_file

    logging.info(f"Copying file: {input_file} to destination: {destination_dir}")
    stime = time.perf_counter()
    os.system(f"rsync -a --progress {input_file} {destination_dir}")
    etime = time.perf_counter()
    logging.info(
        f"Copied file | time (sec): {round(etime - stime, 4)} "
        f"size: {get_file_size(output_file)}"
    )
    return output_file


def copy_dir(input_dir, destination_dir, num_threads):
    # remove the backslash if user added it
    data_name = input_dir.strip("/").split("/")[-1]
    if "SLURM_JOBID" in os.environ:
        destination_dir = get_slurm_dir(destination_dir)
    destination_dir = f"{destination_dir}/{data_name}"
    makedir(destination_dir)
    complete_flag = f"{destination_dir}/copy_complete"
    if PathManager.isfile(complete_flag):
        logging.info(f"Found Data already copied: {destination_dir}...")
        return destination_dir
    logging.info(
        f"Copying {input_dir} to dir {destination_dir} using {num_threads} threads"
    )
    # We have to do multi-threaded rsync to speed up copy.
    cmd = (
        f"ls -d {input_dir}/* | parallel -j {num_threads} --will-cite "
        f"rsync -ruW --inplace {{}} {destination_dir}"
    )
    os.system(cmd)
    PathManager.open(complete_flag, "a").close()
    logging.info("Copied to local directory")
    return destination_dir


def copy_data(input_file, destination_dir, num_threads):
    # return whatever the input is: whether "", None or anything else.
    logging.info(f"Creating directory: {destination_dir}")
    makedir(destination_dir)
    if PathManager.isfile(input_file):
        output_file = copy_file(input_file, destination_dir)
    elif PathManager.isdir(input_file):
        output_file = copy_dir(input_file, destination_dir, num_threads)
    return output_file


def copy_data_to_local(input_files, destination_dir, num_threads=40):
    # it might be possible that we don't use the labels and hence don't have
    # label files. In that case, we return the input_files itself as we have
    # nothing to copy.
    if len(input_files) > 0:
        output_files = []
        for item in input_files:
            if isinstance(item, list):
                copied_file = copy_data_to_local(item, destination_dir)
            else:
                copied_file = copy_data(item, destination_dir, num_threads)
            output_files.append(copied_file)
        return output_files
    return input_files
