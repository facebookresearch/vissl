# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil

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
        help="Path to the folder containing the 'fgvc-aircraft-2013b' folder",
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


def download_fgvc_aircraft(root: str):
    """
    Download the FGVC Aircraft dataset archives and expand them
    in the folder provided as parameter
    """
    url_folder = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
    for url_file in [
        "fgvc-aircraft-2013b.tar.gz",
        "fgvc-aircraft-2013b-annotations.tar.gz",
    ]:
        download_and_extract_archive(url_folder + url_file, root)


def create_fgvc_aircrafts_disk_folder(input_path: str, output_path: str):
    image_folder = os.path.join(input_path, "data", "images")
    label_list = read_label_list(input_path)
    splits = ["trainval", "train", "val", "test"]
    for split in splits:
        print(f"Creating split {split}...")
        labels_path = os.path.join(input_path, "data", f"images_variant_{split}.txt")
        output_split_path = os.path.join(output_path, split)
        for label in label_list:
            os.makedirs(os.path.join(output_split_path, label), exist_ok=True)
        create_fgvc_aircrafts_split(image_folder, labels_path, output_split_path)


def read_label_list(input_path: str):
    variants_file = os.path.join(input_path, "data", "variants.txt")
    with open(variants_file, "r") as f:
        return [aircraft_name_to_label(line.strip()) for line in f]


def aircraft_name_to_label(name: str) -> str:
    return name.replace("/", "-").replace(" ", "-")


def create_fgvc_aircrafts_split(
    image_folder: str, labels_path: str, output_split_path: str
):
    with open(labels_path, "r") as labels_file:
        lines = [line.strip() for line in labels_file]
        for line in tqdm(lines):
            line = line.split(" ")
            image_name = line[0]
            label_name = aircraft_name_to_label(" ".join(line[1:]))
            shutil.copy(
                src=os.path.join(image_folder, f"{image_name}.jpg"),
                dst=os.path.join(output_split_path, label_name),
            )


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_fgvc_aircraft_data_files.py
        -i /path/to/aircrafts/
        -o /output_path/aircrafts
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_fgvc_aircraft(args.input)
    input_path = os.path.join(args.input, "fgvc-aircraft-2013b")
    create_fgvc_aircrafts_disk_folder(input_path=input_path, output_path=args.output)
