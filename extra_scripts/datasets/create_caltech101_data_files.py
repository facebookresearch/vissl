# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil

import numpy as np
from torchvision.datasets.utils import extract_archive
from tqdm import tqdm
from vissl.utils.download import (
    download_google_drive_url,
    get_redirected_url,
    to_google_drive_download_url,
)
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
        help="Path to the folder containing the 101_ObjectCategories folder",
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
        help="To download the original dataset and decompress it in the input folder",
    )
    return parser


def download_caltech_101(root: str):
    """
    Download the FOOD101 dataset archive and expand it in the folder provided as parameter
    """
    url = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
    output_file_name = "101_ObjectCategories.tar.gz"
    url = get_redirected_url(url)
    url = to_google_drive_download_url(url)

    download_google_drive_url(
        url=url, output_path=root, output_file_name=output_file_name
    )
    extract_archive(os.path.join(root, output_file_name), root)


def create_caltech_101_disk_folder(
    input_path: str, output_path: str, num_train_sample: int = 30
):
    """
    Following the VTAB protocol, we create a train and test split of
    such that:
    - 30 images of each cateogy end up in the training set
    - the remaining images end up in the test set
    """
    np.random.seed(0)
    labels = os.listdir(input_path)
    for label in tqdm(labels):
        label_path = os.path.join(input_path, label)
        file_names = sorted(os.listdir(label_path))
        train_indices = set(
            np.random.choice(len(file_names), size=num_train_sample, replace=False)
        )
        train_output_path = os.path.join(output_path, "train", label)
        test_output_path = os.path.join(output_path, "test", label)
        os.makedirs(train_output_path, exist_ok=True)
        os.makedirs(test_output_path, exist_ok=True)
        for sample_index, file_name in enumerate(file_names):
            destination = (
                train_output_path if sample_index in train_indices else test_output_path
            )
            output_file_name = _add_missing_extension(file_name)
            shutil.copy(
                src=os.path.join(label_path, file_name),
                dst=os.path.join(destination, output_file_name),
            )


def _add_missing_extension(file_name: str) -> str:
    if not file_name.endswith(".jpg"):
        return file_name + ".jpg"
    return file_name


def cleanup_unused_files(output_path: str):
    """
    Cleanup the unused folders, as the data now exists in the VISSL compatible format.
    """
    for file_to_delete in ["101_ObjectCategories", "101_ObjectCategories.tar.gz"]:
        file_to_delete = os.path.join(output_path, file_to_delete)
        cleanup_dir(file_to_delete)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_caltech101_data_files.py
        -i /path/to/caltech101/
        -o /output_path/caltech101
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_caltech_101(args.input)
    input_path = os.path.join(args.input, "101_ObjectCategories")
    create_caltech_101_disk_folder(input_path=input_path, output_path=args.output)

    if args.download:
        cleanup_unused_files(args.output)
