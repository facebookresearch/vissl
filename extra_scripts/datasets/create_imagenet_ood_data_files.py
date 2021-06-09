# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
import torchvision.datasets as datasets
from fvcore.common.file_io import PathManager
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Folder containing the imagenet-a and imagenet-r folders",
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
        help="To download the original dataset in the input folder",
    )
    return parser


def download_datasets(root: str):
    """
    Download the Imagenet-A and Imagenet-R dataset archives and expand them
    in the folder provided as parameter
    """
    IMAGENET_A_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
    IMAGENET_R_URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    download_and_extract_archive(url=IMAGENET_A_URL, download_root=root)
    download_and_extract_archive(url=IMAGENET_R_URL, download_root=root)


class ImagenetTargetMapper:
    """
    Use to map the classes of datasets featuring a sub-set of Imagenet classes
    such as Imagenet-A or Imagenet-R to the Imagenet classes index
    """

    IMAGENET_TARGETS_URL = (
        "https://dl.fbaipublicfiles.com/vissl/data/imagenet_classes.txt"
    )

    def __init__(self):
        with PathManager.open(self.IMAGENET_TARGETS_URL) as f:
            imagenet_classes = [line.strip() for line in f.readlines()]
            imagenet_classes.sort()
        self.label_to_id = {label: i for i, label in enumerate(imagenet_classes)}

    def get_target_mapping(self, path: str):
        path_classes = [
            p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))
        ]
        path_classes.sort()
        return {i: self.label_to_id[c] for i, c in enumerate(path_classes)}


def create_imagenet_test_files(input_path: str, output_path: str):
    """
    Create a test split for ImageNet, by reading the image folder in the
    'input_path' and mapping its targets to the targets of imagenet
    """

    # Map the images of the image folder to their corresponding targets
    # in ImageNet
    image_paths, image_labels = [], []
    target_mapper = ImagenetTargetMapper()
    target_mappings = target_mapper.get_target_mapping(input_path)
    input_dataset = datasets.ImageFolder(root=input_path, loader=lambda x: x)
    for image_path, target_id in tqdm(input_dataset):
        image_paths.append(image_path)
        image_labels.append(target_mappings[target_id])

    # Save the these lists in the disk_filelist format
    os.makedirs(output_path, exist_ok=True)
    img_info_out_path = os.path.join(output_path, "test_images.npy")
    label_info_out_path = os.path.join(output_path, "test_labels.npy")
    np.save(img_info_out_path, np.array(image_paths))
    np.save(label_info_out_path, np.array(image_labels))


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_imagenet_ood_data_files.py
        -i /path/to/imagenet_ood/
        -o /output_path/to/imagenet_ood
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_datasets(args.input)

    for dataset_name in ["imagenet-a", "imagenet-r"]:
        input_path = os.path.join(args.input, dataset_name)
        if os.path.exists(input_path):
            output_path = os.path.join(args.output, dataset_name)
            create_imagenet_test_files(input_path, output_path)
