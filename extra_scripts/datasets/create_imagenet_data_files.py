# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script can be used to extract the ImageNet-1K and Places205 dataset files
[data, labels] from the given data source path. The files can be prepared for
both data splits train/val.
Credits: https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/extra_scripts/create_imagenet_data_files.py  # noqa
"""

import argparse
import logging
import os
import sys

import numpy as np
from iopath.common.file_io import g_pathmgr


# initiate the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_all_classes(data_dir):
    class_idx = {}
    counter = 0
    for item in os.listdir(data_dir):
        if item not in ["." ".."]:
            class_idx[item] = counter
            counter += 1
    return class_idx


def get_images_labels_info(split, args):
    assert g_pathmgr.exists(args.data_source_dir), "Data source NOT found. Abort!"
    data_dir = f"{args.data_source_dir}/{split}"
    class_idx = get_all_classes(data_dir)
    logger.info("Number of classes in {} data: {}".format(split, len(class_idx)))
    all_classes = class_idx.keys()
    image_paths, image_classes, img_ids = [], [], []
    for class_name in all_classes:
        class_label = class_idx[class_name]
        class_dir = f"{data_dir}/{class_name}"
        # get all the images in this dir
        for item in os.listdir(class_dir):
            if item not in [".", ".."]:
                image_paths.append(f"{class_dir}/{item}")
                img_ids.append(f"{class_name}/{item}")
                image_classes.append(class_label)
    output_dict = {}
    if args.generate_json:
        for idx in range(len(img_ids)):
            id = img_ids[idx]
            lbl = image_classes[idx]
            output_dict[id] = lbl
    return image_paths, image_classes, output_dict


def main():
    parser = argparse.ArgumentParser(
        description="Create the ImageNet/Places205 data information file"
    )
    parser.add_argument(
        "--data_source_dir",
        type=str,
        default=None,
        help="Path to data directory containing train and val images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where images/label information will be written",
    )
    parser.add_argument(
        "--generate_json",
        action="store_true",
        help="Whether to json files for partitions train/test/val",
    )
    args = parser.parse_args()

    # given the data directory for the partitions train/val both. We will write
    # numpy file since pickle files are prone to failures across different
    # python version
    partitions = ["train", "val"]
    for partition in partitions:
        logger.info(f"========Preparing {partition} data files========")
        imgs_info, lbls_info, output_dict = get_images_labels_info(partition, args)
        img_info_out_path = f"{args.output_dir}/{partition}_images.npy"
        label_info_out_path = f"{args.output_dir}/{partition}_labels.npy"
        logger.info("=================SAVING DATA files=======================")
        logger.info(f"partition: {partition} saving img_paths to: {img_info_out_path}")
        logger.info(f"partition: {partition} saving lbls_paths: {label_info_out_path}")
        logger.info(f"partition: {partition} imgs: {np.array(imgs_info).shape}")

        np.save(img_info_out_path, np.array(imgs_info))
        np.save(label_info_out_path, np.array(lbls_info))
        if args.generate_json:
            json_out_path = f"{args.output_dir}/{partition}_targets.json"
            import json

            with g_pathmgr.open(json_out_path, "w") as fp:
                json.dump(output_dict, fp)
            logger.info("Saved Json to: {}".format(json_out_path))
    logger.info("DONE!")


if __name__ == "__main__":
    main()
