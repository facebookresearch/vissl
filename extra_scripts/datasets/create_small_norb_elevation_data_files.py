# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
from PIL import Image
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
        help="Path to the folder containing the original small NORM decompressed archives",
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
    Download the CLEVR dataset archive and expand it in the folder provided as parameter
    """
    URLS = [
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz",
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
        "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz",
    ]
    for url in URLS:
        download_and_extract_archive(url=url, download_root=root)


def parse_small_norb_format(raw_content):
    """
    The format is described in https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/.
    Here are the parts that we exploit in this function:

        The header is best described by a C structure:

        struct header {
            int magic; // 4 bytes
            int ndim; // 4 bytes, little endian
            int dim[3];
        };

        When the matrix has more than 3 dimensions, the header will be followed by further dimension size information.

        The magic number encodes the element type of the matrix:
        - 0x1E3D4C51 for a single precision matrix
        - 0x1E3D4C53 for a double precision matrix
        - 0x1E3D4C54 for an integer matrix
        - 0x1E3D4C55 for a byte matrix
        - 0x1E3D4C56 for a short matrix

        The files [..] use the little-endian scheme to encode the 4-byte integers.
        Pay attention when you read the files on machines that use big-endian.
    """
    _DATA_TYPES = {
        0x1E3D4C51: "<f4",
        0x1E3D4C53: "<f8",
        0x1E3D4C54: "<i4",
        0x1E3D4C55: "<u1",
        0x1E3D4C56: "<i2",
    }

    dtype_code = np.frombuffer(raw_content, dtype="<i4", count=1)[0]
    shape_size = np.frombuffer(raw_content, dtype="<i4", count=1, offset=4)[0]
    shape = tuple(np.frombuffer(raw_content, dtype="<i4", count=shape_size, offset=8))
    data_offset = max(5, 2 + shape_size) * 4
    data = np.frombuffer(raw_content, dtype=_DATA_TYPES[dtype_code], offset=data_offset)
    return np.reshape(data, shape)


def read_small_norb_format(file_path: str):
    with open(file_path, "rb") as f:
        return parse_small_norb_format(f.read())


def parse_image_array(dat: np.ndarray):
    return dat[:, 0, :, :]  # Keep only the first image of the image pair


def parse_elevation(info: np.ndarray):
    elevations = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70])
    return elevations[info[:, 1]]


def parse_azimuth(info: np.ndarray):
    return info[:, 2] * 10


def create_disk_folder_split(
    dat_path: str, info_path: str, output_path: str, target_transform
):
    training_images = parse_image_array(read_small_norb_format(dat_path))
    training_target = target_transform(read_small_norb_format(info_path))
    with tqdm(total=training_target.shape[0]) as progress_bar:
        for i in range(training_target.shape[0]):
            image = Image.fromarray(training_images[i], mode="L").convert(mode="RGB")
            elevation = str(training_target[i])
            image_folder = os.path.join(output_path, elevation)
            os.makedirs(image_folder, exist_ok=True)
            image.save(os.path.join(image_folder, f"image_{i}.jpg"))
            progress_bar.update(1)


def create_norm_elevation_dataset(input_path: str, output_path: str, target_transform):
    create_disk_folder_split(
        dat_path=os.path.join(
            input_path, "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat"
        ),
        info_path=os.path.join(
            input_path, "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat"
        ),
        output_path=os.path.join(output_path, "train"),
        target_transform=target_transform,
    )
    create_disk_folder_split(
        dat_path=os.path.join(
            input_path, "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat"
        ),
        info_path=os.path.join(
            input_path, "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat"
        ),
        output_path=os.path.join(output_path, "test"),
        target_transform=target_transform,
    )


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_small_norb_elevation_data_files.py -i /path/to/small_norb/ -o /output_path/to/small_norb
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    create_norm_elevation_dataset(
        input_path=args.input, output_path=args.output, target_transform=parse_elevation
    )
