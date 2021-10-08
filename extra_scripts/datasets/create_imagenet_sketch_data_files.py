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
from torchvision.datasets.utils import extract_archive
from vissl.utils.download import download_google_drive_url
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
    Download the Imagenet-Sketch dataset archives and expand them
    in the folder provided as parameter.
    """
    url = "https://drive.google.com/uc?export=download&id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA"
    output_file_name = "ImageNet-Sketch.zip"
    download_google_drive_url(
        url=url, output_path=root, output_file_name=output_file_name
    )
    extract_archive(os.path.join(root, output_file_name), root)


def cleanup_unused_files(output_path: str):
    """
    Cleanup the unused folders, as the data now exists in the VISSL compatible format.
    """
    for file_to_delete in ["ImageNet-Sketch.zip", "images"]:
        file_to_delete = os.path.join(output_path, file_to_delete)
        cleanup_dir(file_to_delete)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/create_imagenet_sketch_data_files.py
        -i /path/to/imagenet_sketch/
        -o /output_path/to/imagenet_sketch
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_datasets(args.input)

    input_path = os.path.join(args.input, "imagenet_sketch")
    assert g_pathmgr.exists(input_path), "Input data path does not exist"
    remove_file_name_whitespace(input_path)
    create_imagenet_test_files(input_path, args.output)

    if args.download:
        cleanup_unused_files(args.output)
