import argparse
import os
import pathlib
import shutil

from tqdm import tqdm


def file_len(fname):
    with open(fname) as f:
        for _i, _l in enumerate(f):
            pass
    return _i + 1


if __name__ == '__main__':
    """Creates a subset of imagenet data for fine-tuning (a lá
    https://github.com/google-research/simclr) using the ImageNet train data
    directory and a text file indicating the images to use. The text file is
    formatted such that each file name is on its own line, e.g.
    n04235860_14959.JPEG
    n04239074_7724.JPEG"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_source', type=str,
                        help='path to imagenet training data directory')
    parser.add_argument('--destination', type=str, help='path to directory in which '
                                                        'subset dataset will '
                                                        'be created')
    parser.add_argument('--subset_file', type=str,
                        default='./imagenet_10percent.txt',
                        help='path to .txt file containing subset')
    # parser.add_argument('--symlink', type=bool, default=True, help='Create '
    #                                                                'symlinks '
    #                                                                'if True, '
    #                                                                'copy '
    #                                                                'image '
    #                                                                'files if '
    #                                                                'False')

    # Parse arguments
    args = parser.parse_args()
    # Ensure subset_path is absolute
    subset_file = os.path.abspath(args.subset_file)
    # Get name of subset
    subset_name = os.path.basename(subset_file).split('.')[0]
    # Generate subset directory string
    subset_data_path = os.path.join(args.destination, subset_name, 'train')
    # Create subset directory
    pathlib.Path(subset_data_path).mkdir(parents=True, exist_ok=True)

    f_len = file_len(subset_file)
    print(f'Copying image files from {args.imagenet_source} to '
          f'{subset_data_path}')
    with tqdm(total=f_len) as pbar:
        with open(subset_file) as reader:
            for line in reader:
                line = line.rstrip()
                image_class = line.split('_')[0]
                # Path to source image
                source_path = os.path.join(args.imagenet_source, image_class, line)
                # Path to destination directory
                destination_directory_path = os.path.join(subset_data_path, image_class)
                # Create destination directory
                pathlib.Path(destination_directory_path).mkdir(parents=True, exist_ok=True)
                # Path to destination image
                destination_file_path = os.path.join(destination_directory_path, line)
                # Copy image file
                shutil.copyfile(source_path, destination_file_path)
                pbar.update()
