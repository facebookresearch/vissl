# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
- This script creates .npy filelist for all splits in the input path.
The script assumes that the input directory has the structure of
the torchvision ImageFolder.
- The script also works if the data doesn't have any splits.

Example usage:

```
python extra_scripts/convert_folder_to_filelist.py \
    -i "manifold://ssl_framework/tree/datasets/food_101/" \
    -o "manifold://ssl_framework/tree/datasets/food_101/"
```

"""

import argparse
import os

from iopath.common.file_io import g_pathmgr
from vissl.utils.env import setup_path_manager
from vissl.utils.io import save_file


def get_filelist_labels_images_paths(input_path):
    dataset_split_summary = {}
    img_paths, img_labels = [], []
    label_paths = g_pathmgr.ls(input_path)
    dataset_split_summary["labels"] = label_paths
    dataset_split_summary["num_labels"] = len(label_paths)
    print(f"{len(label_paths)} classes found.")

    total_split_examples = 0
    # Populate the img_paths and img_labels based on torchvision image folder file structure.
    for label in label_paths:
        label_path = os.path.join(input_path, label)
        images = g_pathmgr.ls(os.path.join(input_path, label))
        print(f"{len(images)} examples found for {label}.")
        total_split_examples += len(images)
        for image in images:
            img_path = os.path.join(label_path, image)
            img_paths.append(img_path)
            img_labels.append(label)

    # print the dataset summary
    dataset_split_summary["num_examples"] = total_split_examples
    print(f"{total_split_examples} found")
    return dataset_split_summary, img_paths, img_labels


def save_img_labels_filelist(
    img_paths, img_labels, out_image_filepath, out_label_filepath
):
    # Remove the split .npy filelist if they exist and resave them.
    if g_pathmgr.exists(out_image_filepath):
        g_pathmgr.rm(out_image_filepath)
    save_file(img_paths, out_image_filepath)
    print(f"Saved: {out_image_filepath}")

    if g_pathmgr.exists(out_label_filepath):
        g_pathmgr.rm(out_label_filepath)
    save_file(img_labels, out_label_filepath)
    print(f"Saved: {out_label_filepath}")
    print("Saved!!")


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Root Directory with train/test/val paths.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Directory to put test.npy, train.npy, and/or val.npy.",
    )
    return parser


if __name__ == "__main__":
    """
    Example usage:

    python extra_scripts/convert_folder_to_filelist.par \
        -i "manifold://ssl_framework/tree/datasets/food_101/" \
        -o "manifold://ssl_framework/tree/datasets/food_101/"
    """
    args = get_argument_parser().parse_args()

    setup_path_manager()

    ground_truth_splits = ["train", "trainval", "val", "test"]
    available_splits = g_pathmgr.ls(args.input)

    dataset_summary = {}

    if not any(split in available_splits for split in ground_truth_splits):
        # the dataset doesn't have any splits. So we just read it as is
        print("Dataset has no splits...")
        dataset_summary, img_paths, img_labels = get_filelist_labels_images_paths(
            args.input
        )
        out_image_filepath = os.path.join(args.output, "images.npy")
        out_label_filepath = os.path.join(args.output, "labels.npy")
        save_img_labels_filelist(
            img_paths, img_labels, out_image_filepath, out_label_filepath
        )
    else:
        for split in ["train", "trainval", "val", "test"]:
            if not g_pathmgr.exists(os.path.join(args.input, split)):
                continue

            dataset_summary[split] = {}
            split_path = os.path.join(args.input, split)
            print(f"Getting data filelist for split: {split}")
            (
                dataset_summary[split],
                img_paths,
                img_labels,
            ) = get_filelist_labels_images_paths(split_path)

            # Remove the split .npy filelist if they exist and resave them..
            out_image_filepath = os.path.join(args.output, f"{split}_images.npy")
            out_label_filepath = os.path.join(args.output, f"{split}_labels.npy")
            save_img_labels_filelist(
                img_paths, img_labels, out_image_filepath, out_label_filepath
            )

    # Save dataset summary.
    dataset_summary_path = os.path.join(args.output, "dataset_summary.json")
    if g_pathmgr.exists(dataset_summary_path):
        g_pathmgr.rm(dataset_summary_path)
    save_file(dataset_summary, dataset_summary_path)
