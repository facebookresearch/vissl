# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import ssl
from contextlib import contextmanager
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url, extract_archive
from tqdm import tqdm


try:
    from pyunpack import Archive
except ImportError:
    raise ValueError(
        "You must have pyunpack and patool installed to run this script: pip install pyunpack patool."
    )

try:
    import av
except ImportError:
    raise ValueError("You must have pyav installed to run this script: pip install av.")


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The input folder contains the expanded UCF-101 archive files",
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


@contextmanager
def without_ssl_certificate_check():
    default_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    yield
    ssl._create_default_https_context = default_context


def download_dataset(root: str):
    """
    Download the UCF101 dataset archive and expand it in the folder provided as parameter
    """
    IMAGE_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    SPLIT_URL = (
        "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
    )

    # Download the raw inputs of UCF101, circumventing the SSL certificate issues
    with without_ssl_certificate_check():
        download_url(url=IMAGE_URL, root=root)
        download_url(url=SPLIT_URL, root=root)

    # Extract the archives
    print("Extracting archives...")
    Archive(os.path.join(root, "UCF101.rar")).extractall(root)
    extract_archive(os.path.join(root, "UCF101TrainTestSplits-RecognitionTask.zip"))


class _ExtractMiddleFrameDataset:
    """
    Dataset used to parallelize the transformation of the dataset via a DataLoader
    """

    def __init__(self, data_path: str, annotation_path: str):
        self.data_path = data_path
        self.split_info = self._read_split_info(annotation_path)

    @staticmethod
    def _read_split_info(file_path: str) -> List[Tuple[str, str]]:
        samples = []
        with open(file_path) as f:
            for line in f:
                category, file_name = line.strip().split("/")
                file_name = file_name.split(" ")[0]
                samples.append((category, file_name))
        return samples

    @staticmethod
    def _extract_middle_frame(file_path: str) -> Optional[Image.Image]:
        """
        Extract the middle frame out of a video clip.
        """
        with av.open(file_path) as container:
            nb_frames = container.streams.video[0].frames
            vid_stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(vid_stream)):
                if i - 1 == nb_frames // 2:
                    return frame.to_image()
            return None

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, str]:
        category, video_name = self.split_info[idx]
        video_path = os.path.join(self.data_path, category, video_name)
        mid_frame = self._extract_middle_frame(video_path)
        image_name = os.path.splitext(video_name)[0] + ".jpg"
        return mid_frame, image_name, category


def create_disk_folder_split(annotation_path: str, data_path: str, output_path: str):
    """
    Create one split of the disk_folder format from the file at 'annotation_path' and the data stored
    in the folder 'data_path'.
    """
    assert os.path.exists(
        annotation_path
    ), f"Could not find annotation path {annotation_path}"
    assert os.path.exists(data_path), f"Could not find data folder {data_path}"

    dataset = _ExtractMiddleFrameDataset(
        data_path=data_path, annotation_path=annotation_path
    )
    loader = DataLoader(dataset, num_workers=8, batch_size=1, collate_fn=lambda x: x[0])
    for batch in tqdm(loader):
        mid_frame, image_name, category = batch
        category_folder = os.path.join(output_path, category)
        os.makedirs(category_folder, exist_ok=True)
        image_path = os.path.join(category_folder, image_name)
        with open(image_path, "w") as image_file:
            mid_frame.save(image_file)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_ucf101_data_files.py -i /path/to/ucf101 -o /output_path/ucf101 -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)

    data_path = os.path.join(args.input, "UCF-101")
    annotation_path = os.path.join(args.input, "ucfTrainTestlist")
    create_disk_folder_split(
        annotation_path=os.path.join(annotation_path, "trainlist01.txt"),
        data_path=data_path,
        output_path=os.path.join(args.output, "train"),
    )
    create_disk_folder_split(
        annotation_path=os.path.join(annotation_path, "testlist01.txt"),
        data_path=data_path,
        output_path=os.path.join(args.output, "val"),
    )
