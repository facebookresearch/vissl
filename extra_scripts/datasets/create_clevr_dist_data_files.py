# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bisect
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
        help="Path to the folder containing the original CLEVR_v1.0 dataset",
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


def create_clevr_distance_disk_folder(input_path: str, output_path: str):
    """
    Transform the CLEVR_v1.0 dataset in folder 'input_path' to a classifcation dataset
    following the disk_folder format at 'output_path' where the goal is to estimate the
    distance of the closest object.
    """
    thresholds = np.array([8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
    target_labels = [f"below_{threshold}" for threshold in thresholds]

    for split in ("train", "val"):
        print(f"Processing the {split} split...")

        # Read the scene description, holding all object information
        scenes_path = os.path.join(input_path, "scenes", f"CLEVR_{split}_scenes.json")
        with open(scenes_path) as f:
            scenes = json.load(f)["scenes"]

        # Associate the images with their corresponding target
        image_paths = []
        image_labels = []
        for scene in tqdm(scenes):
            image_path = os.path.join(
                input_path, "images", split, scene["image_filename"]
            )
            distance = min(object["pixel_coords"][2] for object in scene["objects"])
            target_id = bisect.bisect_left(thresholds, distance)
            target = target_labels[target_id]
            image_paths.append(image_path)
            image_labels.append(target)

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
    python extra_scripts/datasets/create_clevr_dist_data_files.py -i /path/to/clevr/ \
        -o /output_path/to/clevr_dist
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    input_path = os.path.join(args.input, "CLEVR_v1.0")
    create_clevr_distance_disk_folder(input_path=input_path, output_path=args.output)
