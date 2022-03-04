# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from vissl.utils.download import download_and_extract_archive
from vissl.utils.io import save_file


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing the original CLEVR_v1.0 folder",
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


def download_dataset(root: str):
    """
    Download the iWildCam2020 dataset
    URL taken from:
    https://github.com/p-lambda/wilds/blob/main/wilds/datasets/iwildcam_dataset.py
    """
    URL = "https://worksheets.codalab.org/rest/bundles/0x6313da2b204647e79a14b468131fcd64/contents/blob/"
    download_and_extract_archive(url=URL, download_root=root, filename="archive.tar.gz")


def create_iwilds_cam_disk_filelist(input_path: str, output_path: str):
    meta_data_path = os.path.join(input_path, "metadata.csv")
    image_folder = os.path.join(input_path, "train")
    meta_data = pd.read_csv(meta_data_path)

    splits = sorted(set(meta_data["split"]))
    with tqdm(total=len(meta_data)) as pbar:
        for split in splits:
            image_paths, image_labels = [], []
            split_meta_data = meta_data[meta_data["split"] == split]
            split_meta_data = split_meta_data[["filename", "y"]]
            for _, (file_name, y) in split_meta_data.iterrows():
                image_paths.append(os.path.join(image_folder, file_name))
                image_labels.append(y)
                pbar.update(1)
            image_paths_file = os.path.join(output_path, f"{split}_images.npy")
            image_labels_file = os.path.join(output_path, f"{split}_labels.npy")
            save_file(np.array(image_paths), image_paths_file)
            save_file(np.array(image_labels), image_labels_file)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_iwilds_cam_data_files.py
        -i /path/to/iwilds_cam/
        -o /path/to/iwilds_cam/
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_iwilds_cam_disk_filelist(input_path=args.input, output_path=args.output)
