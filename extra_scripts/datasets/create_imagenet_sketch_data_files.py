# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os

from extra_scripts.datasets.create_imagenet_ood_data_files import (
    create_imagenet_test_files,
)
from vissl.utils.download import download_and_extract_archive


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
    Download the Imagenet-A and Imagenet-R dataset archives and expand them
    in the folder provided as parameter.
    """
    # TODO: Change url.
    IMAGENET_SKETCH_URL = (
        "https://drive.google.com/file/d/1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA/view"
    )
    download_and_extract_archive(url=IMAGENET_SKETCH_URL, download_root=root)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/create_imagenet_ood_data_files.py
        -i /path/to/imagenet_ood/
        -o /output_path/to/imagenet_ood
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_datasets(args.input)

    if os.path.exists(args.input):
        create_imagenet_test_files(args.input, args.output)
