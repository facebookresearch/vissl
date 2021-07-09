# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
from typing import List, Tuple

from tqdm import tqdm
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
        help="Path to the folder containing the GTSRB expanded archives",
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


def download_gtsrb(root: str):
    """
    Download the GTSRB dataset archives and expand them in the folder
    provided as parameter
    """
    url_training = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip"
    url_test = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    url_test_gt = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
    download_and_extract_archive(url_training, download_root=root)
    download_and_extract_archive(url_test, download_root=root)
    download_and_extract_archive(url_test_gt, download_root=root)


def create_gtsrb_disk_folder(input_path: str, output_path: str):
    print("Copying the train split...")
    train_input_path = os.path.join(input_path, "GTSRB", "Training")
    train_output_path = os.path.join(output_path, "train")
    os.makedirs(train_output_path, exist_ok=True)
    for label in tqdm(os.listdir(train_input_path)):
        if os.path.isdir(os.path.join(train_input_path, label)):
            shutil.copytree(
                src=os.path.join(train_input_path, label),
                dst=os.path.join(train_output_path, label),
            )

    print("Creating the test split...")
    test_image_folder = os.path.join(input_path, "GTSRB", "Final_Test", "Images")
    test_images_to_label = read_test_labels(input_path)
    for image_name, label in tqdm(test_images_to_label):
        label_folder = os.path.join(output_path, "test", f"{label:05d}")
        os.makedirs(label_folder, exist_ok=True)
        shutil.copy(
            src=os.path.join(test_image_folder, image_name),
            dst=os.path.join(label_folder, image_name),
        )


def read_test_labels(input_path) -> List[Tuple[str, int]]:
    test_label_path = os.path.join(input_path, "GT-final_test.csv")
    with open(test_label_path, "r") as f:
        image_to_label = []
        for i, line in enumerate(f):
            if i > 0:
                cells = line.strip().split(";")
                image_name = cells[0]
                image_label = int(cells[-1])
                image_to_label.append((image_name, image_label))
        return image_to_label


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_gtsrb_data_files.py
        -i /path/to/gtsrb/
        -o /output_path/gtsrb
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_gtsrb(args.input)
    input_path = args.input
    create_gtsrb_disk_folder(input_path=input_path, output_path=args.output)
