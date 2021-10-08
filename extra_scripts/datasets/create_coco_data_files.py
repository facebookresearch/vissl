# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script can be used to extract the COCO2014 dataset files [data, labels]
from the given annotations that can be used for training. The files can be
prepared for various data splits.
"""

import argparse
import json
import logging
import sys

import numpy as np
from iopath.common.file_io import g_pathmgr
from pycocotools.coco import COCO


# initiate the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


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


def get_imgs_labels_info(split, json_file, args):
    assert g_pathmgr.exists(json_file), "Data source does not exist. Abort"
    json_data = json.load(g_pathmgr.open(json_file, "r"))
    image_index = [x["id"] for x in json_data["images"]]
    coco = COCO(json_file)

    num_cats = len(json_data["categories"])
    logger.info(
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
    prefix = args.train_imgs_path if split == "train" else args.val_imgs_path
    for item in sorted(img_labels_map.keys()):
        img_paths.append(f"{prefix}/{item}")
        img_labels.append(img_labels_map[item])
    return img_paths, img_labels


def main():
    parser = argparse.ArgumentParser(description="Create COCO data files")
    parser.add_argument(
        "--json_annotations_dir",
        type=str,
        default=None,
        help="Path for the json dataset annotations for various partitions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where images/label information will be written",
    )
    parser.add_argument(
        "--train_imgs_path",
        type=str,
        default=None,
        help="Path where training images (.jpg) for COCO2014 are stored",
    )
    parser.add_argument(
        "--val_imgs_path",
        type=str,
        default=None,
        help="Path where val images (.jpg) for COCO2014 are stored",
    )
    args = parser.parse_args()

    # given the data directory for the partitions train, val, minival and
    # valminusminival, we will write numpy files for each partition.
    partitions = ["val", "train", "minival", "valminusminival"]
    for partition in partitions:
        annotation_file = f"{args.json_annotations_dir}/instances_{partition}2014.json"
        logger.info("========Preparing {} data files========".format(partition))
        imgs_info, lbls_info = get_imgs_labels_info(partition, annotation_file, args)
        img_info_out_path = f"{args.output_dir}/{partition}_images.npy"
        label_info_out_path = f"{args.output_dir}/{partition}_labels.npy"
        logger.info("=================SAVING DATA files=======================")
        logger.info(f"partition: {partition} saving img_paths to: {img_info_out_path}")
        logger.info(f"partition: {partition} saving lbls_paths: {label_info_out_path}")
        logger.info(f"partition: {partition} imgs: {np.array(imgs_info).shape}")
        np.save(img_info_out_path, np.array(imgs_info))
        np.save(label_info_out_path, np.array(lbls_info))
    logger.info("DONE!")


if __name__ == "__main__":
    main()
