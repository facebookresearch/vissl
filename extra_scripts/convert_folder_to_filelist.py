# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script creates .npy filelist for all splits in the input path.
The script assumes that the input directory has the structure of
the torchvision ImageFolder.

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

    buck-out/gen/deeplearning/projects/ssl_framework/extra_scripts/fb/convert_folder_to_filelist.par \  # NOQA
        -i "manifold://ssl_framework/tree/datasets/food_101/" \
        -o "manifold://ssl_framework/tree/datasets/food_101/"
    """
    args = get_argument_parser().parse_args()

    setup_path_manager()

    splits = g_pathmgr.ls(args.input)
    print(f"The following splits are found: { ','.join(splits) }")

    dataset_summary = {}

    for split in ["train", "trainval", "val", "test"]:
        if not g_pathmgr.exists(os.path.join(args.input, split)):
            continue

        dataset_summary[split] = {}
        img_paths = []
        img_labels = []

        split_path = os.path.join(args.input, split)
        label_paths = g_pathmgr.ls(split_path)
        dataset_summary[split]["labels"] = label_paths
        dataset_summary[split]["num_labels"] = len(label_paths)
        print(f"{len(label_paths)} classes found for { split } split.")

        total_split_examples = 0
        # Populate the img_paths and img_labels based on torchvision image folder file structure.
        for label in label_paths:
            label_path = os.path.join(split_path, label)
            images = g_pathmgr.ls(os.path.join(split_path, label))
            print(f"{len(images)} examples found for { label }, { split }.")
            total_split_examples += len(images)
            for image in images:
                img_path = os.path.join(label_path, image)
                img_paths.append(img_path)
                img_labels.append(label)

        dataset_summary[split]["num_examples"] = total_split_examples
        print(f"{ total_split_examples } found for { split } split \n")
        # Remove the split .npy filelist if they exist and resave them..
        image_path = os.path.join(args.output, f"{split}_images.npy")

        g_pathmgr.rm(image_path)
        save_file(img_paths, image_path)
        print(f"Saved { image_path }")

        label_path = os.path.join(args.output, f"{split}_labels.npy")

        g_pathmgr.rm(label_path)
        save_file(img_labels, label_path)
        print(f"Saved { label_path }")

    # Save dataset summary.
    dataset_summary_path = os.path.join(args.output, "dataset_summary.json")
    g_pathmgr.rm(dataset_summary_path)
    save_file(dataset_summary, dataset_summary_path)
