"""
This script is used to extract the iNaturalist2018 dataset disk_filelist.
If download flag is specified, dataset will be downloaded to the 'input' path.
Otherwise we will expect the typical iNaturalist2018 dataset structure.
Please see: https://github.com/visipedia/inat_comp/tree/master/2018#data

The script prepares training and validation data and produces
train and validation source/label `.npy` files.
These files will be written to the '-o' path.

Usage Example run from parent directory:
$ python extra_scripts/datasets/create_inaturalist2018_data_files.py \
    -i "/path/to/inaturalist2018/" \
    -o "/output_path/to/inaturalist2018" -d
"""

import argparse
import json
import logging
import sys

import numpy as np
from iopath.common.file_io import g_pathmgr
from vissl.utils.download import download_and_extract_archive
from vissl.utils.io import save_file


# Initiate the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Global Constants
TRAINING_ANNOTATIONS_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz"
)
VAL_ANNOTIATIONS_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz"
)
IMAGES_URL = (
    "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz"
)


def download_dataset(input_dir_path: str):
    """
    Download the iNaturalist2018 annotations and dataset
    """
    logger.info("========Downloading Annotations ========")
    download_and_extract_archive(
        url=TRAINING_ANNOTATIONS_URL, download_root=input_dir_path
    )
    download_and_extract_archive(url=VAL_ANNOTIATIONS_URL, download_root=input_dir_path)

    logger.info("========Downloading Images. This may take awhile! ========")
    download_and_extract_archive(url=IMAGES_URL, download_root=input_dir_path)


def get_images_labels_info(input_file_name: str, input_dir_path: str):
    """
    Process the iNaturalist2018 image paths and labels for the training
    or validation data. This assumes the input_dir_path contains the
    train2018.json and val2018.json files with relative paths to
    training and validation images, respectively, as well as a sub-directory
    with all the raw images
    """
    data_source_path = input_dir_path + input_file_name
    with open(data_source_path) as ds_path:
        data = json.load(ds_path)

    images, labels = [], []

    assert (
        "images" in data
    ), f"Please check the specified {data_source_path} as it doesn't contain 'images' in it"  # noqa
    for info in data["images"]:
        image = info["file_name"]
        path_elements = image.split("/")
        inaturalist_path = "/".join(path_elements[1:])
        full_path = f"{ input_dir_path }/train_val2018/{inaturalist_path}"
        label = int(path_elements[2])
        images.append(full_path)
        labels.append(label)
    return np.array(images), np.array(labels)


def main():
    parser = argparse.ArgumentParser(
        description="Create the iNaturalist2018 data information file."
    )
    parser.add_argument(
        "-i",
        "--input_dir_path",
        type=str,
        help="Path to the parent directory of the iNaturalist2018 data set",
    )
    parser.add_argument(
        "-o",
        "--output_dir_path",
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
    args = parser.parse_args()

    # Make sure that the input and output directories exist.
    assert g_pathmgr.exists(
        args.input_dir_path
    ), "Data input directory not found! Please create the directory"
    assert g_pathmgr.exists(
        args.output_dir_path
    ), "Data output directory not found! Please create the directory"

    # Download dataset to input path
    if args.download:
        download_dataset(args.input_dir_path)

    # Process training and validation datasets into numpy arrays
    logger.info("========Preparing train data files========")
    train_images, train_labels = get_images_labels_info(
        "/train2018.json", args.input_dir_path
    )
    logger.info("========Preparing val data files========")
    val_images, val_labels = get_images_labels_info(
        "/val2018.json", args.input_dir_path
    )

    # Save as numpy files to output path
    logger.info("=================Saving train data files=======================")
    train_label_file_name = f"{ args.output_dir_path }/train_labels.npy"
    train_image_file_name = f"{ args.output_dir_path }/train_images.npy"
    save_file(train_images, train_image_file_name)
    save_file(train_labels, train_label_file_name)

    logger.info("=================Saving val data files=======================")
    val_label_file_name = f"{ args.output_dir_path }/val_labels.npy"
    val_image_file_name = f"{ args.output_dir_path }/val_images.npy"
    save_file(val_images, val_image_file_name)
    save_file(val_labels, val_label_file_name)


if __name__ == "__main__":
    main()
