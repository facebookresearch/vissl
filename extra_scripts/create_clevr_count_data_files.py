import argparse
import json
import os
import shutil

from tqdm import tqdm


def create_clevr_count_disk_folder(input_path: str, output_path: str):
    train_targets = set()
    for split in ("train", "val"):
        print(f"Processing the {split} split...")

        # Read the scene description, holding all object information
        input_image_path = os.path.join(input_path, "images", split)
        output_image_path = os.path.join(output_path, split)
        scenes_path = os.path.join(input_path, "scenes", f"CLEVR_{split}_scenes.json")
        with open(scenes_path) as f:
            scenes = json.load(f)["scenes"]
        image_names = [scene["image_filename"] for scene in scenes]
        targets = [len(scene["objects"]) for scene in scenes]

        # Make sure that the categories in the train and validation sets are the same
        if split == "train":
            train_targets = set(targets)
            print("Number of classes:", len(train_targets))
        else:
            valid_indices = set(i for i in range(len(image_names)) if targets[i] in train_targets)
            image_names = [image_name for i, image_name in enumerate(image_names) if i in valid_indices]
            targets = [target for i, target in enumerate(targets) if i in valid_indices]

        # Create the directories for each target
        for target in train_targets:
            os.makedirs(os.path.join(output_image_path, f"count_{target}"), exist_ok=True)

        # Move the images in their appropriate folder (one folder by target)
        for image_name, target in tqdm(zip(image_names, targets), total=len(targets)):
            shutil.copy(
                src=os.path.join(input_image_path, image_name),
                dst=os.path.join(output_path, split, f"count_{target}", image_name)
            )


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="Path to the folder containing the original CLEVR dataset")
    parser.add_argument('-o', '--output', type=str, help="Folder where the classification dataset will be written")
    return parser


if __name__ == '__main__':
    """
    Example usage:

    ```
    python extra_scripts/create_clevr_count_data_files.py -i /path/to/CLEVR_v1.0/ -o /output_path/clevr_count
    ```
    """
    args = get_argument_parser().parse_args()
    create_clevr_count_disk_folder(input_path=args.input, output_path=args.output)
