# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script can be used to extract the VOC2007 and VOC2012 dataset files
[data, labels] from the given annotations that can be used for training. The
files can be prepared for various data splits
"""

import logging
import os
from glob import glob

import numpy as np
from iopath.common.file_io import g_pathmgr
from vissl.utils.io import makedir, save_file


def get_output_dir():
    curr_folder = os.path.abspath(".")
    datasets_dir = f"{curr_folder}/datasets"
    logging.info(f"Datasets dir: {datasets_dir}")
    makedir(datasets_dir)
    return datasets_dir


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


def get_data_files(split, data_source_dir):
    data_dir = f"{data_source_dir}/ImageSets/Main"
    assert g_pathmgr.exists(data_dir), "Data: {} doesn't exist".format(data_dir)
    test_data_files = glob(os.path.join(data_dir, "*_test.txt"))
    test_data_files = validate_files(test_data_files)
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


def get_voc_images_labels_info(split, data_source_dir):
    assert g_pathmgr.exists(data_source_dir), "Data source NOT found. Abort"
    data_files = get_data_files(split, data_source_dir)
    # we will construct a map for image name to the vector of -1, 0, 1
    # we sort the data_files which gives sorted class names as well
    img_labels_map = {}
    for cls_num, data_path in enumerate(sorted(data_files)):
        # for this class, we have images and each image will have label
        # 1, -1, 0 -> present, not present, ignore respectively as in VOC data.
        with g_pathmgr.open(data_path, "r") as fopen:
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
                    logging.info(
                        "Error processing: {} data_path: {}".format(line, data_path)
                    )
    img_paths, img_labels = [], []
    for item in sorted(img_labels_map.keys()):
        img_paths.append(f"{data_source_dir}/JPEGImages/{item}.jpg")
        img_labels.append(img_labels_map[item])

    # save to the datasets folder and return the path
    output_dir = get_output_dir()
    img_info_out_path = f"{output_dir}/{split}_images.npy"
    label_info_out_path = f"{output_dir}/{split}_labels.npy"
    save_file(np.array(img_paths), img_info_out_path)
    save_file(np.array(img_labels), label_info_out_path)
    return [img_info_out_path, label_info_out_path]
