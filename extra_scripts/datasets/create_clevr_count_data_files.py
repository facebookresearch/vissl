# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os

import numpy as np
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
    Download the CLEVR dataset archive and expand it in the folder provided as parameter
    """
    URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
    download_and_extract_archive(url=URL, download_root=root)


def create_clevr_count_disk_filelist(input_path: str, output_path: str):
    """
    Transform the CLEVR_v1.0 dataset in folder 'input_path' to a classifcation dataset following the
    disk_folder format at 'output_path' where the goal is to count the number of objects in the scene
    """
    train_unique_targets = set()
    for split in ("train", "val"):
        print(f"Processing the {split} split...")

        # Read the scene description, holding all object information
        input_image_path = os.path.join(input_path, "images", split)
        scenes_path = os.path.join(input_path, "scenes", f"CLEVR_{split}_scenes.json")
        with open(scenes_path) as f:
            scenes = json.load(f)["scenes"]
        image_names = [scene["image_filename"] for scene in scenes]
        targets = [len(scene["objects"]) for scene in scenes]

        # Make sure that the categories in the train and validation sets are the same
        # and assigning an identifier to each of the unique target
        if split == "train":
            train_unique_targets = set(targets)
            print("Number of classes:", len(train_unique_targets))
        else:
            valid_indices = {
                i for i in range(len(image_names)) if targets[i] in train_unique_targets
            }
            image_names = [
                image_name
                for i, image_name in enumerate(image_names)
                if i in valid_indices
            ]
            targets = [target for i, target in enumerate(targets) if i in valid_indices]

        # List the images and labels of the partition
        image_paths = []
        image_labels = []
        for image_name, target in tqdm(zip(image_names, targets), total=len(targets)):
            image_paths.append(os.path.join(input_image_path, image_name))
            image_labels.append(f"count_{target}")

        # Save the these lists in the disk_filelist format
        os.makedirs(output_path, exist_ok=True)
        img_info_out_path = os.path.join(output_path, f"{split}_images.npy")
        label_info_out_path = os.path.join(output_path, f"{split}_labels.npy")
        np.save(img_info_out_path, np.array(image_paths))
        np.save(label_info_out_path, np.array(image_labels))


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_clevr_count_data_files.py -i /path/to/clevr/ -o /output_path/to/clevr_count
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    input_path = os.path.join(args.input, "CLEVR_v1.0")
    create_clevr_count_disk_filelist(input_path=input_path, output_path=args.output)
