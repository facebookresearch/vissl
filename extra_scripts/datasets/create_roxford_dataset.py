# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from extra_scripts.datasets.create_rparis_dataset import (
    create_revisited_oxford_paris_dataset,
)
from vissl.utils.download import download_and_extract_archive, download_url
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
        help="Path to the folder containing the data folder",
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


def download_oxford_dataset(root: str):
    """
    Download the Oxford dataset archive and expand it in the folder provided as parameter
    """
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz"
    download_and_extract_archive(images_url, root)

    metadata_url = (
        "http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.pkl"
    )
    download_url(metadata_url, root)


def _add_missing_extension(file_name: str) -> str:
    if not file_name.endswith(".jpg"):
        return file_name + ".jpg"
    return file_name


def cleanup_unused_files(output_path: str):
    """
    Cleanup the unused folders, as the data now exists in the VISSL compatible format.
    """
    file_to_delete = os.path.join(output_path, "oxbuild_images.tgz")
    cleanup_dir(file_to_delete)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_oxford_dataset.py
        -i /path/to/roxford/
        -o /output_path/roxford
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_oxford_dataset(args.input)

    create_revisited_oxford_paris_dataset(
        input_path=args.input, output_path=args.output, dataset_name="roxford5k"
    )

    if args.download:
        cleanup_unused_files(args.output)

    print("Finished!")
