# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil

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
        help="Path to the folder containing the 'images' and 'annotations' folders",
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


def download_oxford_pets(root: str):
    """
    Download the Oxford Pets dataset archives and expand them
    in the folder provided as parameter
    """
    url_folder = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
    for url_file in ["images.tar.gz", "annotations.tar.gz"]:
        download_and_extract_archive(url=url_folder + url_file, download_root=root)


def create_oxford_pets_disk_folder(input_path: str, output_path: str):
    images_path = os.path.join(input_path, "images")
    print("Creating training split...")
    create_oxford_pets_split(
        images_path=images_path,
        annotations_path=os.path.join(input_path, "annotations", "trainval.txt"),
        output_path=os.path.join(output_path, "train"),
    )
    print("Creating test split...")
    create_oxford_pets_split(
        images_path=images_path,
        annotations_path=os.path.join(input_path, "annotations", "test.txt"),
        output_path=os.path.join(output_path, "test"),
    )


def extract_label(image_name: str) -> str:
    image_name = os.path.splitext(image_name)[0]
    return "_".join(image_name.split("_")[:-1])


def create_oxford_pets_split(images_path: str, annotations_path: str, output_path: str):
    with open(annotations_path, "r") as annotations:
        for annotation in tqdm(annotations):
            image_name = annotation.split(" ")[0]
            image_label = extract_label(image_name)
            label_path = os.path.join(output_path, image_label)
            os.makedirs(label_path, exist_ok=True)
            shutil.copy(
                src=os.path.join(images_path, image_name + ".jpg"), dst=label_path
            )


def cleanup_unused_files(output_path: str):
    """
    Cleanup the unused folders, as the data now exists in the VISSL compatible format.
    """
    for file_to_delete in ["annotations", "images"]:
        file_to_delete = os.path.join(output_path, file_to_delete)
        cleanup_dir(file_to_delete)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_oxford_pets_data_files.py
        -i /path/to/pets/
        -o /output_path/pets
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_oxford_pets(args.input)
    create_oxford_pets_disk_folder(input_path=args.input, output_path=args.output)

    if args.download:
        cleanup_unused_files(args.output)
