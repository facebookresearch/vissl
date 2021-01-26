# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Use this script to convert VISSL ResNe(X)ts models to match Torchvision exactly.
"""

import argparse
import logging
import sys

import torch
from fvcore.common.file_io import PathManager
from vissl.utils.checkpoint import replace_module_prefix
from vissl.utils.io import is_url


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# initiate the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def convert_and_save_model(args, replace_prefix):
    assert PathManager.exists(args.output_dir), "Output directory does NOT exist"

    # load the model
    model_path = args.model_url_or_file
    if is_url(model_path):
        logger.info(f"Loading from url: {model_path}")
        model = load_state_dict_from_url(model_path)
    else:
        model = torch.load(model_path, map_location=torch.device("cpu"))

    # get the model trunk to rename
    if "classy_state_dict" in model.keys():
        model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]
    elif "model_state_dict" in model.keys():
        model_trunk = model["model_state_dict"]
    else:
        model_trunk = model
    logger.info(f"Input model loaded. Number of params: {len(model_trunk.keys())}")

    # convert the trunk
    converted_model = replace_module_prefix(model_trunk, "_feature_blocks.")
    logger.info(f"Converted model. Number of params: {len(converted_model.keys())}")

    # save the state
    output_filename = f"converted_vissl_{args.output_name}.torch"
    output_model_filepath = f"{args.output_dir}/{output_filename}"
    logger.info(f"Saving model: {output_model_filepath}")
    torch.save(converted_model, output_model_filepath)
    logger.info("DONE!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert VISSL ResNe(X)ts models to Torchvision"
    )
    parser.add_argument(
        "--model_url_or_file",
        type=str,
        default=None,
        required=True,
        help="Model url or file that contains the state dict",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Output directory where the converted state dictionary will be saved",
    )
    parser.add_argument(
        "--output_name", type=str, default=None, required=True, help="output model name"
    )
    args = parser.parse_args()
    convert_and_save_model(args, replace_prefix="_feature_blocks.")


if __name__ == "__main__":
    main()
