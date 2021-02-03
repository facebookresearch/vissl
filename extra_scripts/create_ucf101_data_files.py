import argparse
import os
from typing import Optional, List, Tuple

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import av
except ImportError:
    raise ValueError("You must have pyav installed to run this script: pip install av.")


def extract_mid_frame(file_path: str) -> Optional[Image.Image]:
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


def extract_split_info(file_path: str) -> List[Tuple[str, str]]:
    """
    Parse an annotation file of ucf101 and list the samples, grouped by category
    """
    samples = []
    with open(file_path) as f:
        for line in f:
            category, file_name = line.strip().split("/")
            file_name = file_name.split(" ")[0]
            samples.append((category, file_name))
    return samples


class _ExtractMiddleFrameDataset:
    """
    Dataset used to parallelize the reading of of the middle frame by using a pytorch DataLoader
    """
    def __init__(self, data_path: str, annotation_path: str):
        self.data_path = data_path
        self.split_info = extract_split_info(annotation_path)

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str, str]:
        category, video_name = self.split_info[idx]
        video_path = os.path.join(self.data_path, video_name)
        mid_frame = extract_mid_frame(video_path)
        image_name = os.path.splitext(video_name)[0] + ".jpg"
        return mid_frame, image_name, category


def create_disk_folder(
    annotation_path: str,
    data_path: str,
    output_path: str,
    num_workers: int,
):
    """
    Read the annotation path, extract from it the split information, and create a folder with the following structure:

        Action1/
            Image1
            ...
        Action2/
            Image2
            ...
    """
    assert os.path.exists(annotation_path), f"Could not find annotation path {annotation_path}"
    assert os.path.exists(data_path), f"Could not find data folder {data_path}"

    dataset = _ExtractMiddleFrameDataset(data_path=data_path, annotation_path=annotation_path)
    loader = DataLoader(dataset, num_workers=num_workers, batch_size=1, collate_fn=lambda x: x[0])
    for batch in tqdm(loader):
        mid_frame, image_name, category = batch
        category_folder = os.path.join(output_path, category)
        os.makedirs(category_folder, exist_ok=True)
        image_path = os.path.join(category_folder, image_name)
        with open(image_path, "w") as image_file:
            mid_frame.save(image_file)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, help="Data path")
    parser.add_argument('-a', '--annotation', type=str, help="File annotation path")
    parser.add_argument('-o', '--output', type=str, help="Output folder")
    parser.add_argument('-w', '--workers', type=int, default=8, help="Number of parallel workers")
    return parser


if __name__ == '__main__':
    """
    Example usage:
    
    ```
    python extra_scripts/create_ucf101_data_files.py \
        -d /datasets01/ucf101/112018/data \
        -a /datasets01/ucf101/112018/ucfTrainTestlist/trainlist01.txt \
        -o /checkpoint/qduval/vissl/ucf101/train
    
    python extra_scripts/create_ucf101_data_files.py \
        -d /datasets01/ucf101/112018/data \
        -a /datasets01/ucf101/112018/ucfTrainTestlist/testlist01.txt \
        -o /checkpoint/qduval/vissl/ucf101/test
    ```
    
    Each of the artifacts pointed by the script correspond to elements which can be downloaded
    directly from the webside of UCF101: https://www.crcv.ucf.edu/data/UCF101.php.
    - the data corresponds to the videos (https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)
    - the annotations corresponds to one of the action recognition train/test split file
      (https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)
    """
    args = get_argument_parser().parse_args()
    create_disk_folder(
        annotation_path=args.annotation,
        data_path=args.data,
        output_path=args.output,
        num_workers=args.workers
    )

