# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is used to convert the checkpoints written by a FSDP model
(checkpoints sharded by GPU) into checkpoints that can be loaded by
FSDP or DDP model running on a different number of GPUs for evaluation.

Example:

    python extra_scripts/convert_sharded_checkpoint.py \
        -i path/to/fsdp_checkpoint.torch \
        -o path/to/eval_checkpoint.torch \
        -t consolidated
"""

import argparse
import enum
import os

from fvcore.common.file_io import PathManager
from hydra.experimental import compose, initialize_config_module
from vissl.config import AttrDict
from vissl.utils.checkpoint import CheckpointFormatConverter
from vissl.utils.env import set_env_vars
from vissl.utils.io import makedir
from vissl.utils.logger import setup_logging, shutdown_logging


class CheckpointType(enum.Enum):
    """
    The list of possible output format for checkpoints:
    - consolidated: contains all the weights of the model
    - sliced: the weights are separated into separate files such
      that each set of parameter is in a separate file, allowing
      to limit the amount of memory used upon loading the checkpoint
    """

    consolidated = enum.auto()
    sliced = enum.auto()


def convert_checkpoint(input_path: str, output_path: str, output_type: str):
    assert PathManager.exists(
        input_path
    ), f"Checkpoint input path: {input_path} not found."

    # Make the output directory if it doesn't exist.
    makedir(os.path.split(output_path)[0])

    setup_logging(__name__)
    if output_type == CheckpointType.consolidated.name:
        CheckpointFormatConverter.sharded_to_consolidated_checkpoint(
            input_path, output_path
        )
    elif output_type == CheckpointType.sliced.name:
        CheckpointFormatConverter.sharded_to_sliced_checkpoint(input_path, output_path)
    shutdown_logging()


def setup_pathmanager():
    """
    Setup PathManager. A bit hacky -- we use the #set_env_vars method to setup pathmanager
    and as such we need to create a dummy config, and dummy values for local_rank and node_id.
    """
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose(
            "defaults",
            overrides=["config=test/integration_test/quick_swav"],
        )
    config = AttrDict(cfg).config
    set_env_vars(local_rank=0, node_id=0, cfg=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to input checkpoint")
    parser.add_argument("-o", "--output", type=str, help="Path to output checkpoint")
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default=CheckpointType.consolidated.name,
        help=f"Output format of the checkpoint ({CheckpointType.consolidated.name}, {CheckpointType.sliced.name})",
    )
    args = parser.parse_args()

    setup_pathmanager()

    convert_checkpoint(args.input, args.output, args.type)
