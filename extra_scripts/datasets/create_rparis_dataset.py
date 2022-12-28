# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil

import numpy as np
from vissl.utils.download import download_and_extract_archive, download_url
from vissl.utils.io import cleanup_dir, load_file, save_file


# Dataset has corrupted files. See https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/
CORRUPTED_FILES = [
    "paris_louvre_000136.jpg",
    "paris_louvre_000146.jpg",
    "paris_moulinrouge_000422.jpg",
    "paris_museedorsay_001059.jpg",
    "paris_notredame_000188.jpg",
    "paris_pantheon_000284.jpg",
    "paris_pantheon_000960.jpg",
    "paris_pantheon_000974.jpg",
    "paris_pompidou_000195.jpg",
    "paris_pompidou_000196.jpg",
    "paris_pompidou_000201.jpg",
    "paris_pompidou_000467.jpg",
    "paris_pompidou_000640.jpg",
    "paris_sacrecoeur_000299.jpg",
    "paris_sacrecoeur_000330.jpg",
    "paris_sacrecoeur_000353.jpg",
    "paris_triomphe_000662.jpg",
    "paris_triomphe_000833.jpg",
    "paris_triomphe_000863.jpg",
    "paris_triomphe_000867.jpg",
]


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing the data folder",
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


def download_paris_dataset(root: str):
    """
    Download the Paris dataset archive and expand it in the folder provided as parameter
    """
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz"
    download_and_extract_archive(images_url, root)

    images_url = "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz"
    download_and_extract_archive(images_url, root)

    # Flatten dir structure
    paris_img_dir = os.path.join(root, "paris")
    for file_type in os.listdir(paris_img_dir):
        file_type_path = os.path.join(paris_img_dir, file_type)
        for img in os.listdir(file_type_path):
            from_path = os.path.join(file_type_path, img)
            shutil.move(from_path, root)

    metadata_url = (
        "http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.pkl"
    )
    download_url(metadata_url, root)


def create_revisited_oxford_paris_dataset(
    input_path: str, output_path: str, dataset_name="rparis6k"
):
    """
    Following the VTAB protocol, we create a train and test split of
    such that:
    - 30 images of each cateogy end up in the training set
    - the remaining images end up in the test set
    """
    # Make database and query paths.
    database_path = os.path.join(output_path, "database")
    query_path = os.path.join(output_path, "queries")
    os.makedirs(database_path, exist_ok=True)
    os.makedirs(query_path, exist_ok=True)

    # Create .npy filelists for all splits.
    cfg = load_file(os.path.join(output_path, f"gnd_{dataset_name}.pkl"))
    queries = np.array(cfg["qimlist"])
    database = np.array(cfg["imlist"])

    print(f"{len(queries)} images in queries")
    print(f"{len(database)} images in database")

    query_filelist, query_labels = [], []
    for i, query in enumerate(queries):
        # Move file to query path.
        from_filepath = os.path.join(input_path, f"{query}.jpg")
        to_filepath = os.path.join(output_path, "queries", f"{query}.jpg")
        shutil.move(from_filepath, to_filepath)

        # Create filelist of image paths.
        query_filelist.append(to_filepath)

        # Create easy, medium, and hard splits as per: https://arxiv.org/abs/1803.11285
        labels = {
            "easy": cfg["gnd"][i]["easy"],
            "easy_images": database[cfg["gnd"][i]["easy"]],
            "hard": cfg["gnd"][i]["hard"],
            "hard_images": database[cfg["gnd"][i]["hard"]],
            "junk": cfg["gnd"][i]["junk"],
            "junk_images": database[cfg["gnd"][i]["junk"]],
            "bbx": cfg["gnd"][i]["bbx"],
        }
        query_labels.append(labels)

    query_images_path = os.path.join(output_path, "query_images.npy")
    save_file(
        query_filelist,
        query_images_path,
    )
    query_labels_path = os.path.join(output_path, "query_labels.npy")
    save_file(
        query_labels,
        query_labels_path,
    )

    database_filelist = []
    for database_img in database:
        # Move file to query path.
        from_filepath = os.path.join(input_path, f"{database_img}.jpg")
        to_filepath = os.path.join(output_path, "database", f"{database_img}.jpg")
        shutil.move(from_filepath, to_filepath)

        # Create database split
        database_filelist.append(to_filepath)

    database_images_path = os.path.join(output_path, "database_images.npy")
    save_file(database_filelist, database_images_path)


def _add_missing_extension(file_name: str) -> str:
    if not file_name.endswith(".jpg"):
        return file_name + ".jpg"
    return file_name


def cleanup_unused_files(output_path: str):
    """
    Cleanup the unused folders, as the data now exists in the VISSL compatible format.
    """
    file_to_delete = os.path.join(output_path, "paris_1.tgz")
    cleanup_dir(file_to_delete)

    file_to_delete = os.path.join(output_path, "paris_2.tgz")
    cleanup_dir(file_to_delete)

    for corrupted_file in CORRUPTED_FILES:
        file_to_delete = os.path.join(output_path, corrupted_file)
        cleanup_dir(file_to_delete)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_paris_dataset.py
        -i /path/to/rparis/
        -o /output_path/rparis
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_paris_dataset(args.input)

    create_revisited_oxford_paris_dataset(
        input_path=args.input, output_path=args.output, dataset_name="rparis6k"
    )

    if args.download:
        cleanup_unused_files(args.output)
