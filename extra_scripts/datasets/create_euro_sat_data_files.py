# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil

from torch.utils.data import DataLoader
from tqdm import tqdm
from vissl.utils.download import download_and_extract_archive


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing expanded EuroSAT.zip archive",
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
    Download the EuroSAT dataset archive and expand it in the folder provided as parameter
    """
    URL = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
    download_and_extract_archive(url=URL, download_root=root)


class _EuroSAT:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    IMAGE_FOLDER = "2750"
    TRAIN_SAMPLES = 1000
    VALID_SAMPLES = 500

    def __init__(self, input_path: str, output_path: str, train: bool):
        self.input_path = input_path
        self.output_path = output_path
        self.train = train
        self.image_folder = os.path.join(self.input_path, self.IMAGE_FOLDER)
        self.images = []
        self.targets = []
        self.labels = sorted(os.listdir(self.image_folder))

        # There is no train/val split in the EUROSAT dataset, so we have to create it
        for i, label in enumerate(self.labels):
            label_path = os.path.join(self.image_folder, label)
            files = sorted(os.listdir(label_path))
            if train:
                self.images.extend(files[: self.TRAIN_SAMPLES])
                self.targets.extend([i] * self.TRAIN_SAMPLES)
            else:
                self.images.extend(
                    files[self.TRAIN_SAMPLES : self.TRAIN_SAMPLES + self.VALID_SAMPLES]
                )
                self.targets.extend([i] * self.VALID_SAMPLES)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx: int) -> bool:
        image_name = self.images[idx]
        target = self.labels[self.targets[idx]]
        image_path = os.path.join(self.image_folder, target, image_name)
        split_name = "train" if self.train else "val"
        shutil.copy(
            image_path, os.path.join(self.output_path, split_name, target, image_name)
        )
        return True


def create_disk_folder_split(dataset: _EuroSAT, split_path: str):
    """
    Create one split (example: "train" or "val") of the disk_folder hierarchy
    """
    for label in dataset.labels:
        os.makedirs(os.path.join(split_path, label), exist_ok=True)
    loader = DataLoader(dataset, num_workers=8, batch_size=1, collate_fn=lambda x: x[0])
    with tqdm(total=len(dataset)) as progress_bar:
        for _ in loader:
            progress_bar.update(1)


def create_euro_sat_disk_folder(input_path: str, output_path: str):
    """
    Read the EUROSAT dataset at 'input_path' and transform it to a disk folder at 'output_path'
    """
    print("Creating the training split...")
    create_disk_folder_split(
        dataset=_EuroSAT(input_path, output_path=output_path, train=True),
        split_path=os.path.join(output_path, "train"),
    )
    print("Creating the validation split...")
    create_disk_folder_split(
        dataset=_EuroSAT(input_path, output_path=output_path, train=False),
        split_path=os.path.join(output_path, "val"),
    )


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_ucf101_data_files.py -i /path/to/euro_sat \
        -o /output_path/to/euro_sat -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_euro_sat_disk_folder(args.input, args.output)
