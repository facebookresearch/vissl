# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import dataclasses
import os
import random
from typing import Any, List

import numpy as np
from iopath.common.file_io import g_pathmgr
from tqdm import tqdm
from vissl.utils.download import download_and_extract_archive


@dataclasses.dataclass
class SplitData:
    """
    Data structure holding the samples associated to a given split
    """

    image_paths: List[str] = dataclasses.field(default_factory=list)
    image_labels: List[int] = dataclasses.field(default_factory=list)


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


def split_sample_list(samples: List[Any]):
    """
    Split a list of samples in train/val/test splits
    """
    random.shuffle(samples)
    num_samples = len(samples)
    val_start = int(round(num_samples * TRAIN_SPLIT_SIZE))
    val_length = int(round(num_samples * VALID_SPLIT_SIZE))
    val_end = val_start + val_length
    return {
        "train": samples[:val_start],
        "val": samples[val_start:val_end],
        "test": samples[val_end:],
        "trainval": samples[:val_end],
    }


def create_sun397_disk_filelist_dataset(input_path: str, output_path: str, seed: int):
    """
    Create partitions "train", "trainval", "val", "test" from the input path of SUN397
    by allocating 70% of labels to "train", 10% to "val" and 20% to "test".
    """
    random.seed(seed)
    g_pathmgr.mkdirs(output_path)

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
        with g_pathmgr.open(image_output_path, mode="wb") as f:
            np.save(f, np.array(samples.image_paths))
        label_output_path = os.path.join(output_path, f"{split}_labels.npy")
        with g_pathmgr.open(label_output_path, mode="wb") as f:
            np.save(f, np.array(samples.image_labels))


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_sun397_data_files.py -i /path/to/sun397/ -o /output_path/to/sun397 -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_sun397_disk_filelist_dataset(
        input_path=args.input, output_path=args.output, seed=args.seed
    )
