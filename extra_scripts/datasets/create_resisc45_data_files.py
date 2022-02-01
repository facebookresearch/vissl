# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import shutil

from torch.utils.data import DataLoader
from tqdm import tqdm


RESISC45_URL = "https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs"


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the expanded NWPU-RESISC45.rar archive (download from: {})".format(
            RESISC45_URL
        ),
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


class _RESISC45:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    TRAIN_SPLIT_PERCENT = 0.8
    TEST_SPLIT_PERCENT = 0.2

    def __init__(self, input_path: str, output_path: str, train: bool):
        self.input_path = input_path
        self.output_path = output_path
        self.train = train
        self.images = []
        self.targets = []
        self.labels = sorted(os.listdir(self.input_path))
        split_generator = random.Random(42)

        # There is no train/val split in the RESISC45 dataset, so we have to create it
        for i, label in enumerate(self.labels):
            print(f"{i} -> {label}")
            label_path = os.path.join(self.input_path, label)
            files = sorted(os.listdir(label_path))
            split_generator.shuffle(files)
            train_samples = int(self.TRAIN_SPLIT_PERCENT * len(files))
            test_samples = int(self.TEST_SPLIT_PERCENT * len(files))
            if train:
                self.images.extend(files[:train_samples])
                self.targets.extend([i] * train_samples)
            else:
                self.images.extend(files[train_samples : train_samples + test_samples])
                self.targets.extend([i] * test_samples)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int) -> bool:
        image_name = self.images[idx]
        target = self.labels[self.targets[idx]]
        image_path = os.path.join(self.input_path, target, image_name)
        split_name = "train" if self.train else "test"
        shutil.copy(
            image_path, os.path.join(self.output_path, split_name, target, image_name)
        )
        return True


def create_disk_folder_split(dataset: _RESISC45, split_path: str):
    """
    Create one split (example: "train" or "test") of the disk_folder hierarchy
    """
    for label in dataset.labels:
        os.makedirs(os.path.join(split_path, label), exist_ok=True)
    loader = DataLoader(dataset, num_workers=8, batch_size=1, collate_fn=lambda x: x[0])
    with tqdm(total=len(dataset)) as progress_bar:
        for _ in loader:
            progress_bar.update(1)


def create_resisc_disk_folder(input_path: str, output_path: str):
    """
    Read the RESISC45 dataset at 'input_path' and transform it to a disk folder at 'output_path'
    """
    print("Creating the training split...")
    create_disk_folder_split(
        dataset=_RESISC45(input_path, output_path=output_path, train=True),
        split_path=os.path.join(output_path, "train"),
    )
    print("Creating the validation split...")
    create_disk_folder_split(
        dataset=_RESISC45(input_path, output_path=output_path, train=False),
        split_path=os.path.join(output_path, "test"),
    )


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_resisc45_data_files.py \
        -i /path/to/resisc45 -o /output_path/to/resisc45
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        raise Exception(
            f"Cannot automatically download RESISC45. "
            f"You can manually download the archive at {RESISC45_URL}"
        )
    create_resisc_disk_folder(args.input, args.output)
