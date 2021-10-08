# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from extra_scripts.datasets.create_imagenet_a_data_files import (
    create_imagenet_test_files,
    remove_file_name_whitespace,
)
from iopath.common.file_io import g_pathmgr
from vissl.utils.download import download_and_extract_archive
from vissl.utils.io import cleanup_dir


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Folder containing the imagenet-a and imagenet-r folders",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Folder where the classification dataset will be written",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_const",
        const=True,
        default=False,
        help="To download the original dataset in the input folder",
    )
    return parser


def download_datasets(root: str):
    """
    Download the Imagenet-R dataset archives and expand them
    in the folder provided as parameter
    """
    IMAGENET_R_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    download_and_extract_archive(url=IMAGENET_R_URL, download_root=root)


def cleanup_unused_files(output_path: str):
    """
    Cleanup the unused folders, as the data now exists in the VISSL compatible format.
    """
    file_to_delete = os.path.join(output_path, "imagenet-r.tar")
    cleanup_dir(file_to_delete)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_imagenet_r_data_files.py
        -i /path/to/imagenet_r/
        -o /output_path/to/imagenet_r
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_datasets(args.input)

    dataset_name = "imagenet-r"
    input_path = os.path.join(args.input, dataset_name)
    assert g_pathmgr.exists(input_path), "Input data path does not exist"
    remove_file_name_whitespace(input_path)
    create_imagenet_test_files(input_path, args.output)

    if args.download:
        cleanup_unused_files(args.output)
