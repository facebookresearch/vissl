# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from PIL import Image
from torch.utils.data import DataLoader
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
        help="Path to the folder containing the original FOOD-101 dataset",
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
    Download the FOOD101 dataset archive and expand it in the folder provided as parameter
    """
    URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    download_and_extract_archive(url=URL, download_root=root)


class Food101:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    META_FOLDER = "meta"
    IMAGE_FOLDER = "images"
    IMAGE_EXT = ".jpg"

    def __init__(self, input_path: str, output_path: str, split: str):
        self.input_path = input_path
        self.output_path = output_path
        self.split = split
        self.class_file = os.path.join(self.input_path, self.META_FOLDER, "classes.txt")
        self.split_path = os.path.join(
            self.input_path, self.META_FOLDER, split.lower() + ".txt"
        )
        self.IMAGE_FOLDER = os.path.join(self.input_path, self.IMAGE_FOLDER)
        with open(self.class_file, "r") as f:
            self.classes = {line.strip() for line in f}

        self.targets = []
        self.images = []
        with open(self.split_path, "r") as f:
            for line in f:
                label, image_file_name = line.strip().split("/")
                assert label in self.classes, f"Invalid label: {label}"
                self.targets.append(label)
                self.images.append(
                    os.path.join(
                        self.input_path,
                        self.IMAGE_FOLDER,
                        label,
                        image_file_name + self.IMAGE_EXT,
                    )
                )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image_name = os.path.split(image_path)[1]
        image = Image.open(image_path)
        if image.mode == "L":
            image = image.convert("RGB")
        target = self.targets[idx]
        image.save(os.path.join(self.output_path, self.split, target, image_name))
        return True


def create_food_101_disk_folder(input_path: str, output_path: str):
    for split in ["train", "test"]:
        dataset = Food101(input_path=input_path, output_path=output_path, split=split)
        for label in dataset.classes:
            os.makedirs(os.path.join(output_path, split, label), exist_ok=True)
        loader = DataLoader(
            dataset, num_workers=8, batch_size=1, collate_fn=lambda x: x[0]
        )
        with tqdm(total=len(dataset)) as progress_bar:
            for _ in loader:
                progress_bar.update(1)


def cleanup_unused_files(output_path: str):
    """
    Cleanup the unused folders, as the data now exists in the VISSL compatible format.
    """
    for file_to_delete in ["food-101", "food-101.tar.gz"]:
        file_to_delete = os.path.join(output_path, file_to_delete)
        cleanup_dir(file_to_delete)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_food101_data_files.py -i /path/to/food101/ \
        -o /output_path/food101
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    input_path = os.path.join(args.input, "food-101")
    create_food_101_disk_folder(input_path=input_path, output_path=args.output)

    if args.download:
        cleanup_unused_files(args.output)
