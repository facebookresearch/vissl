# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script can be used to extract the VOC2007 and VOC2012 dataset files
[data, labels] from the given annotations that can be used for training. The
files can be prepared for various data splits
"""

import argparse
import logging
import os
import sys
from glob import glob

import numpy as np
from fvcore.common.file_io import PathManager


# initiate the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def validate_files(input_files):
    """
    The valid files will have name: <class_name>_<split>.txt. We want to remove
    all the other files from the input.
    """
    output_files = []
    for item in input_files:
        if len(item.split("/")[-1].split("_")) == 2:
            output_files.append(item)
    return output_files


def get_data_files(split, args):
    data_dir = f"{args.data_source_dir}/ImageSets/Main"
    assert PathManager.exists(data_dir), "Data: {} doesn't exist".format(data_dir)
    test_data_files = glob(os.path.join(data_dir, "*_test.txt"))
    test_data_files = validate_files(test_data_files)
    if args.separate_partitions > 0:
        train_data_files = glob(os.path.join(data_dir, "*_train.txt"))
        val_data_files = glob(os.path.join(data_dir, "*_val.txt"))
        train_data_files = validate_files(train_data_files)
        val_data_files = validate_files(val_data_files)
        assert len(train_data_files) == len(val_data_files)
        if split == "train":
            data_files = train_data_files
        elif split == "test":
            data_files = test_data_files
        else:
            data_files = val_data_files
    else:
        train_data_files = glob(os.path.join(data_dir, "*_trainval.txt"))
        if len(test_data_files) == 0:
            # For VOC2012 dataset, we have trainval, val and train data.
            train_data_files = glob(os.path.join(data_dir, "*_train.txt"))
            test_data_files = glob(os.path.join(data_dir, "*_val.txt"))
        test_data_files = validate_files(test_data_files)
        train_data_files = validate_files(train_data_files)
        data_files = train_data_files if (split == "train") else test_data_files
    assert len(train_data_files) == len(test_data_files), "Missing classes"
    return data_files


def get_images_labels_info(split, args):
    assert PathManager.exists(args.data_source_dir), "Data source NOT found. Abort"

    data_files = get_data_files(split, args)
    # we will construct a map for image name to the vector of -1, 0, 1
    # we sort the data_files which gives sorted class names as well
    img_labels_map = {}
    for cls_num, data_path in enumerate(sorted(data_files)):
        # for this class, we have images and each image will have label
        # 1, -1, 0 -> present, not present, ignore respectively as in VOC data.
        with PathManager.open(data_path, "r") as fopen:
            for line in fopen:
                try:
                    img_name, orig_label = line.strip().split()
                    if img_name not in img_labels_map:
                        img_labels_map[img_name] = -(
                            np.ones(len(data_files), dtype=np.int32)
                        )
                    orig_label = int(orig_label)
                    # in VOC data, -1 (not present), set it to 0 as train target
                    if orig_label == -1:
                        orig_label = 0
                    # in VOC data, 0 (ignore), set it to -1 as train target
                    elif orig_label == 0:
                        orig_label = -1
                    img_labels_map[img_name][cls_num] = orig_label
                except Exception:
                    logger.info(
                        "Error processing: {} data_path: {}".format(line, data_path)
                    )

    img_paths, img_labels = [], []
    for item in sorted(img_labels_map.keys()):
        img_paths.append(f"{args.data_source_dir}/JPEGImages/{item}.jpg")
        img_labels.append(img_labels_map[item])

    output_dict = {}
    if args.generate_json:
        cls_names = []
        for item in sorted(data_files):
            name = item.split("/")[-1].split(".")[0].split("_")[0]
            cls_names.append(name)

        img_ids, json_img_labels = [], []
        for item in sorted(img_labels_map.keys()):
            img_ids.append(item)
            json_img_labels.append(img_labels_map[item])

        for img_idx in range(len(img_ids)):
            img_id = img_ids[img_idx]
            out_lbl = {}
            for cls_idx in range(len(cls_names)):
                name = cls_names[cls_idx]
                out_lbl[name] = int(json_img_labels[img_idx][cls_idx])
            output_dict[img_id] = out_lbl
    return img_paths, img_labels, output_dict


def main():
    parser = argparse.ArgumentParser(description="Create VOC data files")
    parser.add_argument(
        "--data_source_dir",
        type=str,
        default=None,
        help="Path to data directory containing ImageSets and JPEGImages",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where images/label information will be written",
    )
    parser.add_argument(
        "--separate_partitions",
        type=int,
        default=0,
        help="Whether to create files separately for partitions train/test/val",
    )
    parser.add_argument(
        "--generate_json",
        type=int,
        default=0,
        help="Whether to json files for partitions train/test/val",
    )
    args = parser.parse_args()

    # given the data directory for the partitions train, val, and test, we will
    # write numpy files for each partition.
    partitions = ["train", "test"]
    if args.separate_partitions > 0:
        partitions.append("val")

    for partition in partitions:
        logger.info("========Preparing {} data files========".format(partition))
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

            with PathManager.open(json_out_path, "w") as fp:
                json.dump(output_dict, fp)
            logger.info("Saved Json to: {}".format(json_out_path))
    logger.info("DONE!")


if __name__ == "__main__":
    main()
