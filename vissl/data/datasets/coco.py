# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script can be used to extract the COCO2014 dataset files [data, labels]
from the given annotations that can be used for training. The files can be
prepared for various data splits.
"""

import json
import logging
import os

import numpy as np
from fvcore.common.file_io import PathManager

# COCO API
from pycocotools.coco import COCO
from vissl.utils.io import makedir, save_file


def get_output_dir():
    curr_folder = os.path.abspath(".")
    datasets_dir = f"{curr_folder}/datasets"
    logging.info(f"Datasets dir: {datasets_dir}")
    makedir(datasets_dir)
    return datasets_dir


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1.0, np.maximum(0.0, x1))
    y1 = np.minimum(height - 1.0, np.maximum(0.0, y1))
    x2 = np.minimum(width - 1.0, np.maximum(0.0, x2))
    y2 = np.minimum(height - 1.0, np.maximum(0.0, y2))
    return x1, y1, x2, y2


def get_valid_objs(entry, objs):
    valid_objs = []
    width, height = entry["width"], entry["height"]
    for obj in objs:
        if "ignore" in obj and obj["ignore"] == 1:
            continue
        # Convert form x1, y1, w, h to x1, y1, x2, y2
        x1 = obj["bbox"][0]
        y1 = obj["bbox"][1]
        x2 = x1 + np.maximum(0.0, obj["bbox"][2] - 1.0)
        y2 = y1 + np.maximum(0.0, obj["bbox"][3] - 1.0)
        x1, y1, x2, y2 = clip_xyxy_to_image(x1, y1, x2, y2, height, width)
        # Require non-zero seg area and more than 1x1 box size
        if obj["area"] > 0 and x2 > x1 and y2 > y1:
            valid_objs.append(obj)
    return valid_objs


def get_coco_imgs_labels_info(split, data_source_dir, args):
    json_file = f"{data_source_dir}/annotations/instances_{split}2014.json"
    assert PathManager.exists(json_file), "Annotations file does not exist. Abort"
    json_data = json.load(PathManager.open(json_file, "r"))
    image_index = [x["id"] for x in json_data["images"]]
    coco = COCO(json_file)

    num_cats = len(json_data["categories"])
    logging.info(
        "partition: {} num_cats: {} num_images: {}".format(
            split, num_cats, len(image_index)
        )
    )
    cat_ids = [x["id"] for x in json_data["categories"]]
    coco_to_me = {val: ind for ind, val in enumerate(cat_ids)}
    cat_names = [str(x["name"]) for x in json_data["categories"]]
    cat_name_to_id, cat_id_to_name = {}, {}
    for ind, name in enumerate(cat_names):
        cat_name_to_id[name] = ind
        cat_id_to_name[ind] = name

    class_ids = cat_id_to_name.keys()
    assert len(list(class_ids)) == num_cats
    assert min(class_ids) == 0
    assert max(class_ids) == len(class_ids) - 1
    assert len(set(class_ids)) == len(class_ids)
    # label_matrix = np.zeros((len(image_index), len(cat_names)), dtype=np.float32)
    # area_matrix = np.zeros((len(image_index), len(cat_names)), dtype=np.float32)
    img_labels_map = {}
    num_classes = len(cat_names)
    for _, im_id in enumerate(image_index):
        ann_ids = coco.getAnnIds(imgIds=im_id)
        entry = coco.imgs[im_id]
        img_name = entry["file_name"]
        objs = coco.loadAnns(ann_ids)
        valid_objs = get_valid_objs(entry, objs)
        if img_name not in img_labels_map:
            img_labels_map[img_name] = np.zeros(num_classes, dtype=np.int32)
        for _, obj in enumerate(valid_objs):
            cocoCatId = obj["category_id"]
            myId = coco_to_me[cocoCatId]
            img_labels_map[img_name][myId] = 1.0

    # label = 1 (present), 0 (not present)
    img_paths, img_labels = [], []
    train_imgs_path = f"{data_source_dir}/train2014"
    val_imgs_path = f"{data_source_dir}/val2014"
    prefix = train_imgs_path if split == "train" else val_imgs_path
    for item in sorted(img_labels_map.keys()):
        img_paths.append(f"{prefix}/{item}")
        img_labels.append(img_labels_map[item])

    # save to the datasets folder and return the path
    output_dir = get_output_dir()
    img_info_out_path = f"{output_dir}/{split}_images.npy"
    label_info_out_path = f"{output_dir}/{split}_labels.npy"
    save_file(np.array(img_paths), img_info_out_path)
    save_file(np.array(img_labels), label_info_out_path)
    return [img_info_out_path, label_info_out_path]
