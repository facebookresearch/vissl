# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
A script to extract the momentum encoder from DINO as a separate
evaluable checkpoint (to easily benchmark DINO momentum encoder)
"""

import argparse

import torch
from iopath.common.file_io import g_pathmgr
from vissl.utils.checkpoint import CheckpointItemType, DINOCheckpointUtils
from vissl.utils.env import setup_path_manager


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    return parser.parse_args()


def extract_momentum_encoder(input_path: str, output_path: str):
    # Load the checkpoint
    setup_path_manager()
    with g_pathmgr.open(input_path, "rb") as f:
        input_cp = torch.load(f, map_location="cpu")

    # Dispatch the extraction depending on checkpoint type
    checkpoint_type = input_cp.get("type", CheckpointItemType.consolidated.name)
    if checkpoint_type == CheckpointItemType.consolidated.name:
        DINOCheckpointUtils.extract_teacher_from_consolidated_checkpoint(
            input_cp, output_path
        )
    elif checkpoint_type == CheckpointItemType.shard_list.name:
        DINOCheckpointUtils.extract_teacher_from_sharded_checkpoint(
            input_path, output_path
        )


if __name__ == "__main__":
    args = parse_arguments()
    extract_momentum_encoder(input_path=args.input, output_path=args.output)
