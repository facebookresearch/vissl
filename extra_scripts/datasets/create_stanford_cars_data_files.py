# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil

from scipy import io
from torch.utils.data import DataLoader
from tqdm import tqdm
from vissl.utils.download import download_and_extract_archive, download_url
from vissl.utils.io import cleanup_dir


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing the expanded archives from http://imagenet.stanford.edu/internal/car196",
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
        help="To download the original dataset and decompress it in the input folder",
    )
    return parser


STANFORD_URL = "https://ai.stanford.edu/~jkrause/car196/"
TRAIN_IMAGE_URL = STANFORD_URL + "cars_train.tgz"
TRAIN_ANNOT_URL = "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
TEST_IMAGE_URL = STANFORD_URL + "cars_test.tgz"
TEST_ANNOT_URL = STANFORD_URL + "cars_test_annos_withlabels.mat"


def download_dataset(root: str):
    """
    Download the Standford Cars dataset archives and expand them in the folder provided as parameter
    """
    download_and_extract_archive(url=TRAIN_IMAGE_URL, download_root=root)
    download_and_extract_archive(url=TRAIN_ANNOT_URL, download_root=root)
    download_and_extract_archive(url=TEST_IMAGE_URL, download_root=root)
    download_url(url=TEST_ANNOT_URL, root=root)


class StanfordCars:
    """
    The StanfordCars dataset, mapping images to their respective class
    """

    def __init__(self, root: str, split: str):
        assert split in {"train", "test"}
        self.root = root
        self.split = split
        self.annotations = self._open_annotations()
        self.image_folder = os.path.join(self.root, f"cars_{split}")
        self.class_names = self._get_class_names()

    def _open_annotations(self):
        annotations = None
        if self.split == "train":
            annotations = io.loadmat(
                os.path.join(self.root, "devkit/cars_train_annos.mat")
            )
        elif self.split == "test":
            annotations = io.loadmat(
                os.path.join(self.root, "cars_test_annos_withlabels.mat")
            )
        return annotations["annotations"][0]

    def _get_class_names(self):
        meta_data = io.loadmat(os.path.join(self.root, "devkit/cars_meta.mat"))
        class_names = meta_data["class_names"][0]
        return [
            # Format class names appropriately for directory creation.
            "{:03}".format(i) + "_" + class_name[0].replace(" ", "_").replace("/", "_")
            for i, class_name in enumerate(class_names)
        ]

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx: int):
        image_name = self.annotations[idx][5][0]
        target_id = self.annotations[idx][4][0, 0]
        image_path = os.path.join(self.image_folder, image_name)
        # Beware: Stanford cars targets starts at 1
        target_name = self.class_names[target_id - 1]
        return image_path, target_name


class StandfordCarsMapper:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    def __init__(self, dataset: StanfordCars, output_path: str):
        self.dataset = dataset
        self.output_path = output_path

    def init_folders(self):
        for class_name in self.dataset.class_names:
            os.makedirs(
                os.path.join(self.output_path, self.dataset.split, class_name),
                exist_ok=True,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> bool:
        image_path, target_name = self.dataset[idx]
        image_name = os.path.split(image_path)[-1]
        shutil.copy(
            image_path,
            os.path.join(self.output_path, self.dataset.split, target_name, image_name),
        )
        return True


def create_dataset(input_folder: str, output_folder: str):
    """
    Read the dSprites dataset and split it into a training split and a validation split
    which follows the disk_folder format of VISSL
    """
    for split in ["train", "test"]:
        print(f"Processing '{split}' split...")
        dataset = StanfordCars(root=input_folder, split=split)
        mapper = StandfordCarsMapper(dataset, output_path=output_folder)
        mapper.init_folders()
        loader = DataLoader(
            mapper, num_workers=8, batch_size=1, collate_fn=lambda x: x[0]
        )
        with tqdm(total=len(dataset)) as progress_bar:
            for _ in loader:
                progress_bar.update(1)


def cleanup_unused_files(output_path: str):
    """
    Cleanup the unused folders, as the data now exists in the VISSL compatible format.
    """
    for file_to_delete in [
        "cars_train",
        "cars_test",
        "car_devkit.tgz",
        "cars_test.tgz",
        "cars_test_annos_withlabels.mat",
        "cars_train.tgz",
    ]:
        file_to_delete = os.path.join(output_path, file_to_delete)
        cleanup_dir(file_to_delete)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_stanford_cars_data_files.py -i /path/to/cars/ -o /output_path/to/cars -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_dataset(input_folder=args.input, output_folder=args.output)

    if args.download:
        cleanup_unused_files(args.output)
