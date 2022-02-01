# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
import math
import os
import shutil
from typing import Set

import numpy as np
from torch.utils.data import DataLoader
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
    Download the KITTI dataset archive and expand it in the folder provided as parameter
    """
    IMAGE_URL = (
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    )
    LABEL_URL = (
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    )
    download_and_extract_archive(url=IMAGE_URL, download_root=root)
    download_and_extract_archive(url=LABEL_URL, download_root=root)


class KITTIDistance:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    THRESHOLDS = np.array([0.0, 8.0, 20.0])
    LABELS = ["no_vehicle", "below_8", "below_20", "above_20"]

    def __init__(self, image_folder: str, annotation_folder: str, output_path: str):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.output_path = output_path
        self.image_names = sorted(os.listdir(image_folder))
        self.annotation_names = sorted(os.listdir(annotation_folder))
        self.nb_samples = len(self.image_names)
        self.training_set_ids = self._get_training_set_ids(ratio=0.8)

    def __len__(self):
        return self.nb_samples

    def __getitem__(self, idx: int) -> bool:
        """
        Read the annotations of the image at index 'idx' to find its target and
        copy it to its associated folder
        """
        image_name = self.image_names[idx]
        distance = self._get_closest_vehicle_distance(self.annotation_names[idx])
        target = self._get_target_bin(distance)
        split = "train" if idx in self.training_set_ids else "val"
        shutil.copy(
            src=os.path.join(self.image_folder, image_name),
            dst=os.path.join(self.output_path, split, target, image_name),
        )
        return True

    def _get_training_set_ids(self, ratio: float) -> Set[int]:
        """
        Extract a training set from the KITTI distance dataset
        """
        permutation = np.random.permutation(self.nb_samples)
        threshold = int(round(self.nb_samples * ratio))
        return set(permutation[:threshold])

    def _get_closest_vehicle_distance(self, annotation_name: str):
        """
        Read the annotation file associated to an image and extract the distance
        to the closest vehicle in the image or "-inf" if no vehicle is found
        """
        vehicle_types = {"Car", "Van", "Truck"}
        with open(
            os.path.join(self.annotation_folder, annotation_name), "r"
        ) as annotations:
            distances = []
            for annotation in annotations:
                annotation = annotation.split()
                object_type = annotation[0]
                object_dist = float(annotation[13])
                if object_type in vehicle_types:
                    distances.append(object_dist)
            return min(distances, default=-math.inf)

    def _get_target_bin(self, distance: float) -> str:
        """
        The distance are separate into 4 bins:
        - one bin when no vehicle is in the image
        - three equally sized bins (with thresholds 8, 20 and inf)
        """
        target = bisect.bisect_left(self.THRESHOLDS, distance)
        return self.LABELS[target]


def create_dataset(image_folder: str, annotation_folder: str, output_path: str):
    """
    Read the training set of KITTI and split it into a training split and a validation split
    which follows the disk_folder format of VISSL
    """
    dataset = KITTIDistance(image_folder, annotation_folder, output_path)
    for split in ["train", "val"]:
        for target in KITTIDistance.LABELS:
            os.makedirs(os.path.join(output_path, split, target), exist_ok=True)

    loader = DataLoader(dataset, num_workers=8, batch_size=1, collate_fn=lambda x: x[0])
    with tqdm(total=len(dataset)) as progress_bar:
        for _ in loader:
            progress_bar.update(1)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_kitti_dist_data_files.py -i /path/to/kitti/ \
        -o /output_path/to/kitti
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_dataset(
        image_folder=os.path.join(args.input, "training", "image_2"),
        annotation_folder=os.path.join(args.input, "training", "label_2"),
        output_path=args.output,
    )
