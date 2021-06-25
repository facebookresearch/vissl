# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import Set

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vissl.utils.download import download_and_extract_archive


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing the original dSprites repository",
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


def download_dataset(root: str):
    """
    Download the dSprites dataset archive and expand it in the folder provided as parameter
    """
    URL = "https://github.com/deepmind/dsprites-dataset/archive/master.zip"
    download_and_extract_archive(url=URL, download_root=root)


class DSprites(Dataset):
    """
    The DSprites dataset, mapping images to the latent vector used to generate the image
    """

    def __init__(self, root: str, target_transform=None):
        npz_file_name = (
            "dsprites-dataset-master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        )
        self.npz_file_path = os.path.join(root, npz_file_name)
        self.npz_file = np.load(self.npz_file_path)
        self.images = self.npz_file["imgs"]
        self.latents = self.npz_file["latents_values"]
        self.target_transform = target_transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        image_array = self.images[idx] * 255
        image = Image.fromarray(image_array, mode="L").convert("RGB")
        target = self.target_transform(self.latents[idx])
        return image, target


class DSpritesMapper(Dataset):
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    def __init__(self, dataset: DSprites, output_path: str):
        self.dataset = dataset
        self.output_path = output_path
        self.training_set_ids = self._get_training_set_ids(ratio=0.8)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> bool:
        image, target = self.dataset[idx]
        split = "train" if idx in self.training_set_ids else "val"
        os.makedirs(
            os.path.join(self.output_path, split, f"bin_{target}"), exist_ok=True
        )
        image.save(
            os.path.join(self.output_path, split, f"bin_{target}", f"image_{idx}.jpg")
        )
        return True

    def _get_training_set_ids(self, ratio: float) -> Set[int]:
        """
        Extract a training set from the dSprites dataset
        """
        nb_samples = len(self.dataset)
        permutation = np.random.permutation(nb_samples)
        threshold = int(round(nb_samples * ratio))
        return set(permutation[:threshold])


def get_binned_x_position(latents) -> int:
    """
    Return X location of the sprite, binned in 16 different classes.

    The original x position is given as a float between 0.0 and 1.0, so
    we transform it by multiplying it by the number of buckets.
    """
    max_value = 31
    original_nb_buckets = 32
    nb_buckets = 16
    return int(np.floor(latents[4] * max_value * (nb_buckets / original_nb_buckets)))


def create_dataset(input_folder: str, output_folder: str, target_transform):
    """
    Read the dSprites dataset and split it into a training split and a validation split
    which follows the disk_folder format of VISSL
    """
    dataset = DSprites(root=input_folder, target_transform=target_transform)
    mapper = DSpritesMapper(dataset, output_path=output_folder)
    loader = DataLoader(mapper, num_workers=8, batch_size=1, collate_fn=lambda x: x[0])
    with tqdm(total=len(dataset)) as progress_bar:
        for _ in loader:
            progress_bar.update(1)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_dsprites_location_data_files.py -i /path/to/dsprites/ -o /output_path/to/dsprites_loc -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_dataset(
        input_folder=args.input,
        output_folder=args.output,
        target_transform=get_binned_x_position,
    )
