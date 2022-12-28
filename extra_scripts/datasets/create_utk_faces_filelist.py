# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script creates .npy filelist for the UTK Faces dataset
https://susanqq.github.io/UTKFace/

The labels of each face image is embedded in the file name,
formated like [age]_[gender]_[race]_[date&time].jpg

[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting
    0: White,
    1: Black,
    2: Asian,
    3: Indian, and
    4: Others (like Hispanic, Latino, Middle Eastern).
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image
    was collected to UTKFace

Example usage:
```
python extra_scripts/fb/create_utk_faces_filelist.py \
    -i "/path/to/utk_faces/images/" \
    -o "/path/to/utk_faces/"
```
"""

import argparse
import os

from fvcore.common.file_io import PathManager
from vissl.utils.env import setup_path_manager
from vissl.utils.io import save_file


GENDER_MAPPING = {
    0: "male",
    1: "female",
}

RACE_MAPPING = {
    0: "white",
    1: "black",
    2: "asian",
    3: "indian",
    4: "others_hispanic_latino_middle_eastern",
}


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Root Directory with images for the UTK-Faces",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Directory to put images.npy and labels.npy."
    )
    return parser


def get_filelist_labels_images_paths(input_path):
    dataset_summary, metadata = {}, {}
    img_paths, gender_labels, race_labels, age_labels = [], [], [], []
    inp_image_names = PathManager.ls(input_path)
    print(f"{len(inp_image_names)} images found.")

    total_examples = 0
    # Populate the img_paths and labels labels based on folder file structure.
    for img_name in inp_image_names:
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(input_path, img_name)
        img_paths.append(img_path)

        img_age = int(str(img_name).split("_")[0])
        img_gender = GENDER_MAPPING[int(str(img_name).split("_")[1])]
        img_race = RACE_MAPPING[int(str(img_name).split("_")[2])]
        # import pdb; pdb.set_trace()
        age_labels.append(img_age)
        gender_labels.append(img_gender)
        race_labels.append(img_race)
        metadata[img_name] = {
            "age": img_age,
            "gender": img_gender,
            "race": img_race,
        }
        total_examples += 1

    # print the dataset summary
    print(f"Dataset has {total_examples} images")
    dataset_summary["num_images"] = total_examples
    return dataset_summary, metadata, img_paths, age_labels, gender_labels, race_labels


def save_filelist_data(input_data, output_filepath):
    # Remove the split .npy filelist if they exist and resave them.
    if PathManager.exists(output_filepath):
        PathManager.rm(output_filepath)
    save_file(input_data, output_filepath)


if __name__ == "__main__":
    """
    Example usage:

    python extra_scripts/fb/create_utk_faces_filelist.py \
    -i "/path/to/utk_faces/images/" \
    -o "/path/to/utk_faces/"
    """
    args = get_argument_parser().parse_args()
    setup_path_manager()

    (
        dataset_summary,
        metadata,
        img_paths,
        age_labels,
        gender_labels,
        race_labels,
    ) = get_filelist_labels_images_paths(args.input)
    out_image_filepath = os.path.join(args.output, "images.npy")
    out_age_label_filepath = os.path.join(args.output, "age_labels.npy")
    out_gender_label_filepath = os.path.join(args.output, "gender_labels.npy")
    out_race_label_filepath = os.path.join(args.output, "race_labels.npy")
    dataset_summary_path = os.path.join(args.output, "dataset_summary.json")
    metadata_summary_path = os.path.join(args.output, "dataset_metadata.json")

    save_filelist_data(img_paths, out_image_filepath)
    save_filelist_data(age_labels, out_age_label_filepath)
    save_filelist_data(gender_labels, out_gender_label_filepath)
    save_filelist_data(race_labels, out_race_label_filepath)
    save_filelist_data(dataset_summary, dataset_summary_path)
    save_filelist_data(metadata, metadata_summary_path)
