# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pickle
import re
import shutil
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml
from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr
from vissl.utils.slurm import get_slurm_dir


def cache_url(url: str, cache_dir: str) -> str:
    """
    This implementation downloads the remote resource and caches it locally.
    The resource will only be downloaded if not previously requested.
    """
    parsed_url = urlparse(url)
    dirname = os.path.join(cache_dir, os.path.dirname(parsed_url.path.lstrip("/")))
    makedir(dirname)
    filename = url.split("/")[-1]
    cached = os.path.join(dirname, filename)
    with file_lock(cached):
        if not os.path.isfile(cached):
            logging.info(f"Downloading {url} to {cached} ...")
            cached = download(url, dirname, filename=filename)
    logging.info(f"URL {url} cached in {cached}")
    return cached


# TODO (prigoyal): convert this into RAII-style API
def create_file_symlink(file1, file2):
    """
    Simply create the symlinks for a given file1 to file2.
    Useful during model checkpointing to symlinks to the
    latest successful checkpoint.
    """
    try:
        if g_pathmgr.exists(file2):
            g_pathmgr.rm(file2)
        g_pathmgr.symlink(file1, file2)
    except Exception as e:
        logging.info(f"Could NOT create symlink. Error: {e}")


def save_file(data, filename, append_to_json=True, verbose=True):
    """
    Common i/o utility to handle saving data to various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    Specifically for .json, users have the option to either append (default)
    or rewrite by passing in Boolean value to append_to_json.
    """
    if verbose:
        logging.info(f"Saving data to file: {filename}")
    file_ext = os.path.splitext(filename)[1]
    if file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "wb") as fopen:
            pickle.dump(data, fopen, pickle.HIGHEST_PROTOCOL)
    elif file_ext == ".npy":
        with g_pathmgr.open(filename, "wb") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        if append_to_json:
            with g_pathmgr.open(filename, "a") as fopen:
                fopen.write(json.dumps(data, sort_keys=True) + "\n")
                fopen.flush()
        else:
            with g_pathmgr.open(filename, "w") as fopen:
                fopen.write(json.dumps(data, sort_keys=True) + "\n")
                fopen.flush()
    elif file_ext == ".yaml":
        with g_pathmgr.open(filename, "w") as fopen:
            dump = yaml.dump(data)
            fopen.write(dump)
            fopen.flush()
    else:
        raise Exception(f"Saving {file_ext} is not supported yet")

    if verbose:
        logging.info(f"Saved data to file: {filename}")


def load_file(filename, mmap_mode=None, verbose=True, allow_pickle=False):
    """
    Common i/o utility to handle loading data from various file formats.
    Supported:
        .pkl, .pickle, .npy, .json
    For the npy files, we support reading the files in mmap_mode.
    If the mmap_mode of reading is not successful, we load data without the
    mmap_mode.
    """
    if verbose:
        logging.info(f"Loading data from file: {filename}")

    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".txt":
        with g_pathmgr.open(filename, "r") as fopen:
            data = fopen.readlines()
    elif file_ext in [".pkl", ".pickle"]:
        with g_pathmgr.open(filename, "rb") as fopen:
            data = pickle.load(fopen, encoding="latin1")
    elif file_ext == ".npy":
        if mmap_mode:
            try:
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np.load(
                        fopen,
                        allow_pickle=allow_pickle,
                        encoding="latin1",
                        mmap_mode=mmap_mode,
                    )
            except ValueError as e:
                logging.info(
                    f"Could not mmap {filename}: {e}. Trying without g_pathmgr"
                )
                data = np.load(
                    filename,
                    allow_pickle=allow_pickle,
                    encoding="latin1",
                    mmap_mode=mmap_mode,
                )
                logging.info("Successfully loaded without g_pathmgr")
            except Exception:
                logging.info("Could not mmap without g_pathmgr. Trying without mmap")
                with g_pathmgr.open(filename, "rb") as fopen:
                    data = np.load(fopen, allow_pickle=allow_pickle, encoding="latin1")
        else:
            with g_pathmgr.open(filename, "rb") as fopen:
                data = np.load(fopen, allow_pickle=allow_pickle, encoding="latin1")
    elif file_ext == ".json":
        with g_pathmgr.open(filename, "r") as fopen:
            data = json.load(fopen)
    elif file_ext == ".yaml":
        with g_pathmgr.open(filename, "r") as fopen:
            data = yaml.load(fopen, Loader=yaml.FullLoader)
    elif file_ext == ".csv":
        with g_pathmgr.open(filename, "r") as fopen:
            data = pd.read_csv(fopen)
    else:
        raise Exception(f"Reading from {file_ext} is not supported yet")
    return data


def abspath(resource_path: str):
    """
    Make a path absolute, but take into account prefixes like
    "http://" or "manifold://"
    """
    regex = re.compile(r"^\w+://")
    if regex.match(resource_path) is None:
        return os.path.abspath(resource_path)
    else:
        return resource_path


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        logging.info(f"Error creating directory: {dir_path}")
    return is_success


def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url


def cleanup_dir(dir):
    """
    Utility for deleting a directory. Useful for cleaning the storage space
    that contains various training artifacts like checkpoints, data etc.
    """
    if os.path.exists(dir):
        logging.info(f"Deleting directory: {dir}")
        shutil.rmtree(dir)
    logging.info(f"Deleted contents of directory: {dir}")


def get_file_size(filename):
    """
    Given a file, get the size of file in MB
    """
    size_in_mb = os.path.getsize(filename) / float(1024**2)
    return size_in_mb


def copy_file(input_file, destination_dir, tmp_destination_dir):
    """
    Copy a given input_file from source to the destination directory.

    Steps:
    1. We use g_pathmgr to extract the data to local path.
    2. we simply move the files from the g_pathmgr cached local directory
       to the user specified destination directory. We use rsync.
       How destination dir is chosen:
            a) If user is using slurm, we set destination_dir = slurm_dir (see get_slurm_dir)
            b) If the local path used by PathManafer is same as the input_file path,
               and the destination directory is not specified, we set
               destination_dir = tmp_destination_dir

    Returns:
        output_file (str): the new path of the file
        destination_dir (str): the destination dir that was actually used
    """
    # we first extract the local path for the files. g_pathmgr
    # determines the local path itself and copies data there.
    logging.info(f"Copying {input_file} to local path...")
    out = g_pathmgr.get_local_path(input_file)
    output_dir = os.path.dirname(out)
    logging.info(f"File coped to: {out}")

    if (out == input_file) and not destination_dir:
        destination_dir = tmp_destination_dir
        logging.info(
            f"The file wasn't copied. Copying again to temp "
            f"destination directory: {destination_dir}"
        )
    # if the user wants to copy the files to a specific location,
    # we simply move the files from the g_pathmgr cached directory
    # to the user specified directory.
    destination_dir = get_slurm_dir(destination_dir)
    if "SLURM_JOBID" in os.environ:
        destination_dir = get_slurm_dir(destination_dir)
    if destination_dir is not None:
        makedir(destination_dir)
        output_file = f"{destination_dir}/{os.path.basename(input_file)}"
        if g_pathmgr.exists(output_file):
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
    """
    Copy contents of one directory to the specified destination directory
    using the number of threads to speed up the copy. When the data is
    copied successfully, we create a copy_complete file in the
    destination_dir folder to mark the completion. If the destination_dir
    folder already exists and has the copy_complete file, we don't
    copy the file.

    useful for copying datasets like ImageNet to speed up dataloader.
    Using 20 threads for imagenet takes about 20 minutes to copy.

    Returns:
        destination_dir (str): directory where the contents were copied
    """
    # remove the backslash if user added it
    data_name = input_dir.strip("/").split("/")[-1]
    if "SLURM_JOBID" in os.environ:
        destination_dir = get_slurm_dir(destination_dir)
    destination_dir = f"{destination_dir}/{data_name}"
    makedir(destination_dir)
    complete_flag = f"{destination_dir}/copy_complete"
    if g_pathmgr.isfile(complete_flag):
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
    g_pathmgr.open(complete_flag, "a").close()
    logging.info("Copied to local directory")
    return destination_dir, destination_dir


def copy_data(input_file, destination_dir, num_threads, tmp_destination_dir):
    """
    Copy data from one source to the other using num_threads. The data to copy
    can be a single file or a directory. We check what type of data and
    call the relevant functions.

    Returns:
        output_file (str): the new path of the data (could be file or dir)
        destination_dir (str): the destination dir that was actually used
    """
    # return whatever the input is: whether "", None or anything else.
    logging.info(f"Creating directory: {destination_dir}")
    if not (destination_dir is None or destination_dir == ""):
        makedir(destination_dir)
    else:
        destination_dir = None
    if g_pathmgr.isfile(input_file):
        output_file, output_dir = copy_file(
            input_file, destination_dir, tmp_destination_dir
        )
    elif g_pathmgr.isdir(input_file):
        output_file, output_dir = copy_dir(input_file, destination_dir, num_threads)
    else:
        raise RuntimeError("The input_file is neither a file nor a directory")
    return output_file, output_dir


def copy_data_to_local(
    input_files, destination_dir, num_threads=40, tmp_destination_dir=None
):
    """
    Iteratively copy the list of data to a destination directory.
    Each data to copy could be a single file or a directory.

    Returns:
        output_file (str): the new path of the file. If there were
                           no files to copy, simply return the input_files
        destination_dir (str): the destination dir that was actually used
    """
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
