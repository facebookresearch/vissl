# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil

from tqdm import tqdm
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
        help="Path to the folder containing the decompressed 'dtd' folder",
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


def download_dtd_dataset(root: str):
    """
    Download the Oxford Pets dataset archives and expand them
    in the folder provided as parameter
    """
    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    download_and_extract_archive(url, root)


def create_dtd_disk_folder(input_path: str, output_path: str):
    images_path = os.path.join(input_path, "images")
    splits = [
        ("train", "labels/train1.txt"),
        ("val", "labels/val1.txt"),
        ("test", "labels/test1.txt"),
        ("trainval", "labels/train1.txt"),
        ("trainval", "labels/val1.txt"),
    ]
    for split_name, split_rel_path in splits:
        create_dtd_split(
            images_path=images_path,
            split_path=os.path.join(input_path, split_rel_path),
            output_path=os.path.join(output_path, split_name),
        )


def create_dtd_split(images_path: str, split_path: str, output_path: str):
    with open(split_path, "r") as split:
        for line in tqdm(split):
            image_relative_path = line.strip()
            label_name = os.path.split(image_relative_path)[0]
            label_path = os.path.join(output_path, label_name)
            os.makedirs(label_path, exist_ok=True)
            shutil.copy(
                src=os.path.join(images_path, image_relative_path), dst=label_path
            )


def cleanup_folders(output_path: str):
    """
    Cleanup the zipped folders, as the data now exists in the VISSL compatible format.
    """
    for file_to_delete in ["dtd", "dtd-r1.0.1.tar.gz"]:
        file_to_delete = os.path.join(output_path, file_to_delete)
        cleanup_dir(file_to_delete)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_dtd_data_files.py
        -i /path/to/dtd/
        -o /output_path/dtd
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dtd_dataset(args.input)

    input_path = os.path.join(args.input, "dtd")
    create_dtd_disk_folder(input_path=input_path, output_path=args.output)

    if args.download:
        cleanup_folders(args.output)
