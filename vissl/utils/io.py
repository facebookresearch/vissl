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


def create_file_symlink(file1, file2):
    try:
        if PathManager.exists(file2):
            PathManager.rm(file2)
        PathManager.symlink(file1, file2)
    except Exception as e:
        logging.info(f"Could NOT create symlink. Error: {e}")


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


def load_file(filename, mmap_mode=None):
    logging.info(f"Loading data from file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with PathManager.open(filename, "rb") as fopen:
            data = pickle.load(fopen, encoding="latin1")
    elif file_ext == ".npy":
        if mmap_mode:
            try:
                with PathManager.open(filename, "rb") as fopen:
                    data = np.load(fopen, encoding="latin1", mmap_mode=mmap_mode)
            except ValueError as e:
                logging.info(
                    f"Could not mmap {filename}: {e}. Trying without PathManager"
                )
                data = np.load(filename, encoding="latin1", mmap_mode=mmap_mode)
                logging.info("Successfully loaded without PathManager")
            except Exception:
                logging.info("Could not mmap without PathManager. Trying without mmap")
                with PathManager.open(filename, "rb") as fopen:
                    data = np.load(fopen, encoding="latin1")
        else:
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


def copy_file(input_file, destination_dir, tmp_destination_dir):
    # we first extract the local path for the files. PathManager
    # determines the local path itself and copies data there.
    logging.info(f"Copying {input_file} to local path...")
    out = PathManager.get_local_path(input_file)
    output_dir = os.path.dirname(out)
    logging.info(f"File coped to: {out}")

    if (out == input_file) and not destination_dir:
        destination_dir = tmp_destination_dir
        logging.info(
            f"The file wasn't copied. Copying again to temp "
            f"destination directory: {destination_dir}"
        )
    # if the user wants to copy the files to a specific location,
    # we simply move the files from the PathManager cached directory
    # to the user specified directory.
    destination_dir = get_slurm_dir(destination_dir)
    if "SLURM_JOBID" in os.environ:
        destination_dir = get_slurm_dir(destination_dir)
    if destination_dir is not None:
        makedir(destination_dir)
        output_file = f"{destination_dir}/{os.path.basename(input_file)}"
        if PathManager.exists(output_file):
            logging.info(f"File already copied: {output_file}")
            return output_file, destination_dir

        logging.info(f"Copying file: {input_file} to destination: {destination_dir}")
        stime = time.perf_counter()
        os.system(f"rsync -a --progress {out} {destination_dir}")
        etime = time.perf_counter()
        logging.info(
            f"Copied file | time (sec): {round(etime - stime, 4)} "
            f"size: {get_file_size(output_file)}"
        )
        return output_file, destination_dir
    else:
        return out, output_dir


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
    return destination_dir, destination_dir


def copy_data(input_file, destination_dir, num_threads, tmp_destination_dir):
    # return whatever the input is: whether "", None or anything else.
    logging.info(f"Creating directory: {destination_dir}")
    if not (destination_dir is None or destination_dir == ""):
        makedir(destination_dir)
    else:
        destination_dir = None
    if PathManager.isfile(input_file):
        output_file, output_dir = copy_file(
            input_file, destination_dir, tmp_destination_dir
        )
    elif PathManager.isdir(input_file):
        output_file, output_dir = copy_dir(input_file, destination_dir, num_threads)
    else:
        raise RuntimeError("The input_file is neither a file nor a directory")
    return output_file, output_dir


def copy_data_to_local(
    input_files, destination_dir, num_threads=40, tmp_destination_dir=None
):
    # it might be possible that we don't use the labels and hence don't have
    # label files. In that case, we return the input_files itself as we have
    # nothing to copy.
    if len(input_files) > 0:
        output_files = []
        for item in input_files:
            if isinstance(item, list):
                copied_file, output_dir = copy_data_to_local(
                    item, destination_dir, num_threads, tmp_destination_dir
                )
            else:
                copied_file, output_dir = copy_data(
                    item, destination_dir, num_threads, tmp_destination_dir
                )
            output_files.append(copied_file)
        return output_files, output_dir
    return input_files, destination_dir
