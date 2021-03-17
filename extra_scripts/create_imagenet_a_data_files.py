# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os
from typing import Dict

import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the folder containing the original imagenet-a folder",
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
        help="To download the original dataset in the input folder",
    )
    return parser


def download_dataset(root: str):
    """
    Download the Imagenet-R dataset archive and expand it
    in the folder provided as parameter
    """
    URL = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
    download_and_extract_archive(url=URL, download_root=root)


def create_imagenet_test_files(
    input_path: str, output_path: str, target_mappings: Dict[int, int]
):
    """
    Create a test split for ImageNet, by reading the image folder in the
    'input_path' and mapping its targets to the targets of imagenet
    using the 'target_mappings' dictionary
    """

    # Map the images of the image folder to their corresponding targets
    # in ImageNet
    image_paths = []
    image_labels = []
    imagenet_r = datasets.ImageFolder(root=input_path, loader=lambda x: x)
    for image_path, target_id in tqdm(imagenet_r):
        image_paths.append(image_path)
        image_labels.append(target_mappings[target_id])

    # Save the these lists in the disk_filelist format
    os.makedirs(output_path, exist_ok=True)
    img_info_out_path = os.path.join(output_path, "test_images.npy")
    label_info_out_path = os.path.join(output_path, "test_labels.npy")
    np.save(img_info_out_path, np.array(image_paths))
    np.save(label_info_out_path, np.array(image_labels))


IMAGENET_A_TARGET_MAPPINGS = {
    0: 6,
    1: 11,
    2: 13,
    3: 15,
    4: 17,
    5: 22,
    6: 23,
    7: 27,
    8: 30,
    9: 37,
    10: 39,
    11: 42,
    12: 47,
    13: 50,
    14: 57,
    15: 70,
    16: 71,
    17: 76,
    18: 79,
    19: 89,
    20: 90,
    21: 94,
    22: 96,
    23: 97,
    24: 99,
    25: 105,
    26: 107,
    27: 108,
    28: 110,
    29: 113,
    30: 124,
    31: 125,
    32: 130,
    33: 132,
    34: 143,
    35: 144,
    36: 150,
    37: 151,
    38: 207,
    39: 234,
    40: 235,
    41: 254,
    42: 277,
    43: 283,
    44: 287,
    45: 291,
    46: 295,
    47: 298,
    48: 301,
    49: 306,
    50: 307,
    51: 308,
    52: 309,
    53: 310,
    54: 311,
    55: 313,
    56: 314,
    57: 315,
    58: 317,
    59: 319,
    60: 323,
    61: 324,
    62: 326,
    63: 327,
    64: 330,
    65: 334,
    66: 335,
    67: 336,
    68: 347,
    69: 361,
    70: 363,
    71: 372,
    72: 378,
    73: 386,
    74: 397,
    75: 400,
    76: 401,
    77: 402,
    78: 404,
    79: 407,
    80: 411,
    81: 416,
    82: 417,
    83: 420,
    84: 425,
    85: 428,
    86: 430,
    87: 437,
    88: 438,
    89: 445,
    90: 456,
    91: 457,
    92: 461,
    93: 462,
    94: 470,
    95: 472,
    96: 483,
    97: 486,
    98: 488,
    99: 492,
    100: 496,
    101: 514,
    102: 516,
    103: 528,
    104: 530,
    105: 539,
    106: 542,
    107: 543,
    108: 549,
    109: 552,
    110: 557,
    111: 561,
    112: 562,
    113: 569,
    114: 572,
    115: 573,
    116: 575,
    117: 579,
    118: 589,
    119: 606,
    120: 607,
    121: 609,
    122: 614,
    123: 626,
    124: 627,
    125: 640,
    126: 641,
    127: 642,
    128: 643,
    129: 658,
    130: 668,
    131: 677,
    132: 682,
    133: 684,
    134: 687,
    135: 701,
    136: 704,
    137: 719,
    138: 736,
    139: 746,
    140: 749,
    141: 752,
    142: 758,
    143: 763,
    144: 765,
    145: 768,
    146: 773,
    147: 774,
    148: 776,
    149: 779,
    150: 780,
    151: 786,
    152: 792,
    153: 797,
    154: 802,
    155: 803,
    156: 804,
    157: 813,
    158: 815,
    159: 820,
    160: 823,
    161: 831,
    162: 833,
    163: 835,
    164: 839,
    165: 845,
    166: 847,
    167: 850,
    168: 859,
    169: 862,
    170: 870,
    171: 879,
    172: 880,
    173: 888,
    174: 890,
    175: 897,
    176: 900,
    177: 907,
    178: 913,
    179: 924,
    180: 932,
    181: 933,
    182: 934,
    183: 937,
    184: 943,
    185: 945,
    186: 947,
    187: 951,
    188: 954,
    189: 956,
    190: 957,
    191: 959,
    192: 971,
    193: 972,
    194: 980,
    195: 981,
    196: 984,
    197: 986,
    198: 987,
    199: 988,
}

if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/create_imagenet_a_data_files.py
        -i /path/to/imagenet_a/
        -o /output_path/to/imagenet_a
        -d
    ```
    """
    args = get_argument_parser().parse_args()
    if args.download:
        download_dataset(args.input)
    input_path = os.path.join(args.input, "imagenet-a")
    create_imagenet_test_files(
        input_path=input_path,
        output_path=args.output,
        target_mappings=IMAGENET_A_TARGET_MAPPINGS,
    )
