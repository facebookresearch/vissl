# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil

import numpy as np
import scipy.io
from tqdm import tqdm
from vissl.utils.download import download_and_extract_archive, download_url


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing the 'jpg', 'imagelabels.mat' and 'setid.mat' files",
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


def download_oxford_flowers(root: str):
    """
    Download the Oxford Pets dataset archives and expand them
    in the folder provided as parameter
    """
    url_folder = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    download_and_extract_archive(url_folder + "102flowers.tgz", download_root=root)
    for url_file in ["imagelabels.mat", "setid.mat"]:
        download_url(url_folder + url_file, root)


def create_oxford_flowers_disk_folder(input_path: str, output_path: str):
    images_path = os.path.join(input_path, "jpg")
    labels_path = os.path.join(input_path, "imagelabels.mat")
    setids_path = os.path.join(input_path, "setid.mat")
    set_ids = scipy.io.loadmat(setids_path)
    labels = scipy.io.loadmat(labels_path)
    splits = [("trnid", "train"), ("valid", "train"), ("tstid", "test")]
    for input_split, output_split in splits:
        print(f"Processing split {input_split} to {output_split}...")
        create_oxford_flowers_split(
            images_path,
            labels["labels"][0],
            image_ids=set_ids[input_split][0],
            output_path=os.path.join(output_path, output_split),
        )


def create_oxford_flowers_split(
    images_path: str, labels: np.array, image_ids: np.array, output_path: str
):
    for image_id in tqdm(image_ids):
        image_name = f"image_{image_id:05d}.jpg"
        image_label = labels[image_id - 1]
        output_label_path = os.path.join(output_path, str(image_label))
        os.makedirs(output_label_path, exist_ok=True)
        shutil.copy(
            src=os.path.join(images_path, image_name),
            dst=os.path.join(output_label_path, image_name),
        )


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_oxford_flowers_data_files.py
        -i /path/to/oxford_flowers/
        -o /output_path/oxford_flowers
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_oxford_flowers(args.input)
    create_oxford_flowers_disk_folder(input_path=args.input, output_path=args.output)
