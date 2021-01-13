# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Use this script to convert PyTorch supervised model weights to the VISSL weights.
This script shows conversion for ResNet50 model. You can modify this to convert
other models.
"""

import argparse
import logging
import sys

import torch
from classy_vision.generic.util import save_checkpoint
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


def convert_and_save_model(args, append_prefix):
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
    converted_model = replace_module_prefix(
        model_trunk, f"{append_prefix}5.", f"{append_prefix}layer4."
    )
    converted_model = replace_module_prefix(
        converted_model, f"{append_prefix}4.", f"{append_prefix}layer3."
    )
    converted_model = replace_module_prefix(
        converted_model, f"{append_prefix}3.", f"{append_prefix}layer2."
    )
    converted_model = replace_module_prefix(
        converted_model, f"{append_prefix}2.", f"{append_prefix}layer1."
    )
    converted_model = replace_module_prefix(
        converted_model, f"{append_prefix}0.1.", f"{append_prefix}bn1."
    )
    converted_model = replace_module_prefix(
        converted_model, f"{append_prefix}0.0.", f"{append_prefix}conv1."
    )

    # get the output state to save
    if "classy_state_dict" in model.keys():
        model["classy_state_dict"]["base_model"]["model"]["trunk"] = converted_model
        state = model
    else:
        state = {"classy_state_dict": converted_model}
    logger.info(f"Converted model. Number of params: {len(converted_model.keys())}")

    # save the state
    output_filename = f"converted_vissl_{args.output_name}.torch"
    output_model_filepath = f"{args.output_dir}/{output_filename}"
    logger.info(f"Saving model: {output_model_filepath}")
    save_checkpoint(args.output_dir, state, checkpoint_file=output_filename)
    logger.info("DONE!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SSLIME model to VISSL for Resnet for torchvision compatibility"
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
    convert_and_save_model(args, append_prefix="_feature_blocks.")


if __name__ == "__main__":
    main()
