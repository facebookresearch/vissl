# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import dataclasses
import os
import random
from typing import List

import numpy as np
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing the original SUN397 folder",
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
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed used to create the train/val/test splits",
    )
    return parser


def download_dataset(root: str):
    """
    Download the SUN397 dataset archive and expand it in the folder provided as parameter
    """
    URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    download_and_extract_archive(url=URL, download_root=root)


TRAIN_SPLIT_SIZE = 0.7
VALID_SPLIT_SIZE = 0.1


@dataclasses.dataclass
class SplitData:
    """
    Data structure holding the samples associated to a given split
    """

    image_paths: List[str] = dataclasses.field(default_factory=list)
    image_labels: List[int] = dataclasses.field(default_factory=list)


def split_sample_list(xs):
    """
    Split a list of samples in train/val/test splits
    """
    random.shuffle(xs)
    n = len(xs)
    val_start = int(round(n * TRAIN_SPLIT_SIZE))
    val_length = int(round(n * VALID_SPLIT_SIZE))
    val_end = val_start + val_length
    return {
        "train": xs[:val_start],
        "val": xs[val_start:val_end],
        "test": xs[val_end:],
        "trainval": xs[:val_end],
    }


def create_sun397_disk_filelist_dataset(input_path: str, output_path: str, seed: int):
    """
    Create partitions "train", "trainval", "val", "test" from the input path of SUN397
    by allocating 70% of labels to "train", 10% to "val" and 20% to "test".
    """
    random.seed(seed)
    os.makedirs(output_path, exist_ok=True)

    # List all the available classes in SUN397 and their path
    image_folder = os.path.join(input_path, "SUN397")
    class_names_file = os.path.join(image_folder, "ClassName.txt")
    class_paths = []
    with open(class_names_file, "r") as f:
        for line in f:
            path = line.strip()
            if path.startswith("/"):
                path = path[1:]
            class_paths.append(path)

    # For each label, split the samples in train/val/test and add them
    # to the list of samples associated to each split
    splits_data = {
        "train": SplitData(),
        "val": SplitData(),
        "test": SplitData(),
        "trainval": SplitData(),
    }
    for i, class_path in tqdm(enumerate(class_paths), total=len(class_paths)):
        full_class_path = os.path.join(image_folder, class_path)
        image_names = os.listdir(full_class_path)
        splits = split_sample_list(image_names)
        for split, images in splits.items():
            for image_name in images:
                image_path = os.path.join(full_class_path, image_name)
                splits_data[split].image_paths.append(image_path)
                splits_data[split].image_labels.append(i)

    # Save each split
    for split, samples in splits_data.items():
        image_output_path = os.path.join(output_path, f"{split}_images.npy")
        label_output_path = os.path.join(output_path, f"{split}_labels.npy")
        np.save(image_output_path, np.array(samples.image_paths))
        np.save(label_output_path, np.array(samples.image_labels))


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/create_sun397_data_files.py -i /path/to/sun397/ -o /output_path/to/sun397 -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_sun397_disk_filelist_dataset(
        input_path=args.input, output_path=args.output, seed=args.seed
    )
