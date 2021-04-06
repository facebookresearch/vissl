"""
This script is used to extract the iNaturalist2018 dataset disk_filelist.
If download flag is specified, the dataset will be downloaded to the 'input' file path. 
Otherwise we will expect the typical iNaturalist2018 dataset structure. 
Please see: https://github.com/visipedia/inat_comp/tree/master/2018#data

The script prepares training and validation data and produces train and validation source/label `.npy` files.
These files will be written to the '-o' path. 

Usage Example:
$ python ./create_inaturalist2018_data_files.py -i "/path/to/inaturalist2018/" -o " /output_path/to/inaturalist2018" -d
"""

import argparse
import json
import logging
import numpy as np
import sys
from torchvision.datasets.utils import download_and_extract_archive

# Initiate the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

def download_dataset(input: str):
    """
    Download the iNaturalist2018 annotations and dataset
    """
    logger.info(f"========Downloading Annotations ========")
    TRAINING_ANNOTATIONS_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz"
    VAL_ANNOTIATIONS_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz"
    download_and_extract_archive(url=TRAINING_ANNOTATIONS_URL, download_root=input)
    download_and_extract_archive(url=VAL_ANNOTIATIONS_URL, download_root=input)

    logger.info(f"========Downloading Images. This may take awhile! ========")
    IMAGES_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz"
    download_and_extract_archive(url=IMAGES_URL, download_root=input)

def get_images_labels_info(input_file_name: str, input: str):
    """
    Process the iNaturalist2018 image paths and labels for the training or validation data
    This assumes the input contains the train2018.json and val2018.json files with relative paths to
    training and validation images, respectively, as well as a sub-directory with all the raw images
    """
    data_source_path = input + input_file_name
    with(open(data_source_path)) as ds_path:
        data = json.load(ds_path)

    images, labels = [], []
    for info in data['images']:
        image = info['file_name']
        path_elements = image.split('/')
        # TODO: Figure out correct directory structure. Tar file structure is different than existing dataset on FAIR. 
        full_path = input + "/train_val2018/" + '/'.join(path_elements[1:])
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
        "--input",
        type=str,
        help="Path to the parent directory of the iNaturalist2018 data set",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Folder where output npy files (that wil be called train|val_images.npy and train|val_labels.npy) are generated",
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

    # Download dataset to input path
    if args.download:
        download_dataset(args.input)

    # Process training and validation datasets into numpy arrays
    logger.info(f"========Preparing train data files========")
    train_images, train_labels = get_images_labels_info("/train2018.json", args.input)
    logger.info(f"========Preparing val data files========")
    val_images, val_labels = get_images_labels_info("/val2018.json", args.input)

    # Save as numpy files to output path
    logger.info("=================Saving numpy train data files=======================")
    train_label_file_name = args.output + "/train_labels.npy"
    train_image_file_name = args.output + "/train_images.npy"
    np.save(train_image_file_name, train_images)
    np.save(train_label_file_name, train_labels)

    logger.info("=================Saving numpy val data files=======================")
    val_label_file_name = args.output + "/val_labels.npy"
    val_image_file_name = args.output + "/val_images.npy"
    np.save(val_image_file_name, val_images)
    np.save(val_label_file_name, val_labels)

if __name__ == "__main__":
    main()
