# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
A script to extract the momentum encoder from DINO as a separate
evaluable checkpoint (to easily benchmark DINO momentum encoder)
"""

import argparse
from typing import List

import torch
from vissl.utils.checkpoint import CheckpointItemType


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    return parser.parse_args()


def remove_prefix(key: str, prefixes: List[str]):
    """
    Remove one of the prefixes provided as parameter
    """
    for prefix in prefixes:
        if key.startswith(prefix):
            return key.replace(prefix, "")
    raise ValueError(f"Expected one prefix to be removed among {prefixes}")


def extract_momentum_encoder(input_path: str, output_path: str):
    input_cp = torch.load(input_path, map_location="cpu")
    loss_cp = input_cp["classy_state_dict"]["loss"]

    trunk_weights = {}
    heads_weights = {}
    for k, v in loss_cp.items():
        if "trunk" in k:
            k = remove_prefix(
                k, ["momentum_teacher.module.trunk.", "momentum_teacher.trunk."]
            )
            trunk_weights[k] = v
        elif "heads" in k:
            k = remove_prefix(
                k, ["momentum_teacher.module.heads.", "momentum_teacher.heads"]
            )
            heads_weights[k] = v

    output_cp = {
        "type": CheckpointItemType.consolidated.name,
        "classy_state_dict": {
            "base_model": {"model": {"trunk": trunk_weights, "heads": heads_weights}}
        },
    }
    torch.save(output_cp, output_path)


if __name__ == "__main__":
    args = parse_arguments()
    extract_momentum_encoder(input_path=args.input, output_path=args.output)
