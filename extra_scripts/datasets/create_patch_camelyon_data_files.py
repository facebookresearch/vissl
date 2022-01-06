# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gzip
import os
from collections import OrderedDict
from typing import NamedTuple

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from vissl.utils.download import download_url


try:
    import h5py
except ImportError:
    raise ValueError(
        "You must have h5py installed to run this script: pip install h5py."
    )


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The input folder contains the Patch Camelyon data files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The output folder containing the disk_folder output",
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


class Split(NamedTuple):
    file: str
    url: str


class PatchCamelyon:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    _FILES = OrderedDict(
        {
            "train_x": Split(
                "camelyonpatch_level_2_split_train_x.h5",
                "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz",
            ),
            "train_y": Split(
                "camelyonpatch_level_2_split_train_y.h5",
                "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz",
            ),
            "valid_x": Split(
                "camelyonpatch_level_2_split_valid_x.h5",
                "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz",
            ),
            "valid_y": Split(
                "camelyonpatch_level_2_split_valid_y.h5",
                "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz",
            ),
        }
    )

    def __init__(
        self, input_path: str, output_path: str, split: str, download: bool = False
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.split = split
        self._download_missing_files(download)
        self.x = self._open_h5_file(self._FILES[f"{split}_x"].file)["x"]
        self.y = self._open_h5_file(self._FILES[f"{split}_y"].file)["y"]

    def _download_missing_files(self, download: bool):
        for file, url in self._FILES.values():
            if not os.path.exists(os.path.join(self.input_path, file)):
                if download:
                    filename = os.path.basename(url)
                    download_url(url=url, root=self.input_path, filename=filename)
                    from_path = os.path.join(self.input_path, filename)
                    to_path = from_path.replace(".gz", "")
                    with gzip.open(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
                        wfh.write(rfh.read())
                else:
                    raise ValueError(f"Missing file {file} in {self.input_path}")

    def _open_h5_file(self, file_name: str):
        return h5py.File(os.path.join(self.input_path, file_name), "r")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> bool:
        img = Image.fromarray(self.x[idx])
        label = self.y[idx][0][0]
        folder = os.path.join(self.output_path, "tumor" if label == 1 else "no_tumor")
        img.save(os.path.join(folder, f"img_{idx}.jpg"))
        return True


def to_disk_folder_split(dataset: PatchCamelyon, output_folder: str, num_workers: int):
    """
    Create one split of the disk_folder format
    """
    os.makedirs(os.path.join(output_folder, "tumor"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "no_tumor"), exist_ok=True)
    loader = DataLoader(
        dataset, num_workers=num_workers, batch_size=1, collate_fn=lambda x: x[0]
    )
    with tqdm(total=len(dataset)) as progress_bar:
        for _ in loader:
            progress_bar.update(1)


def create_data_files(input_path: str, output_path: str, download: bool):
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)
    if download:
        os.makedirs(input_path, exist_ok=True)

    # Create the training split
    output_train_folder = os.path.join(output_path, "train")
    train_set = PatchCamelyon(
        input_path=input_path,
        output_path=output_train_folder,
        split="train",
        download=download,
    )
    to_disk_folder_split(train_set, output_train_folder, num_workers=8)

    # Create the validation split
    output_valid_folder = os.path.join(output_path, "val")
    valid_set = PatchCamelyon(
        input_path=input_path,
        output_path=output_valid_folder,
        split="valid",
        download=False,
    )
    to_disk_folder_split(valid_set, output_valid_folder, num_workers=8)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_patch_camelyon_data_files.py \
        -i /path/to/patch_camelyon \
        -o /output_path/to/patch_camelyon -d
    ```
    """
    args = get_argument_parser().parse_args()
    create_data_files(
        input_path=args.input, output_path=args.output, download=args.download
    )
