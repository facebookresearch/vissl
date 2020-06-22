#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Use this script to convert PyTorch supervised model weights to the VISSL weights.
This script shows conversion for ResNet50 model. You can modify this to convert
other models.
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from vissl.utils.checkpoint import append_module_suffix, replace_module_suffix
from vissl.utils.io import is_url


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# initiate the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def convert_and_save_model(args, append_suffix):
    assert os.path.exists(args.output_dir), "Output directory does NOT exist"

    model_path = args.model_url_or_file
    if is_url(model_path):
        logger.info(f"Loading from url: {model_path}")
        pth_sup_model = load_state_dict_from_url(model_path)
    else:
        # we support loading models from a numpy file containing the dictionary
        if model_path.endswith("npy"):
            pth_sup_model = np.load(model_path, allow_pickle=True)[()]
        else:
            pth_sup_model = torch.load(model_path)
    logger.info(f"Input model loaded. Number of params: {len(pth_sup_model.keys())}")
    pth_sup_model = replace_module_suffix(pth_sup_model, "module.", "")
    pth_sup_model = replace_module_suffix(pth_sup_model, "layer", "")
    pth_sup_model = append_module_suffix(pth_sup_model, append_suffix)
    pth_sup_model = replace_module_suffix(
        pth_sup_model, f"{append_suffix}4.", f"{append_suffix}5."
    )
    pth_sup_model = replace_module_suffix(
        pth_sup_model, f"{append_suffix}3.", f"{append_suffix}4."
    )
    pth_sup_model = replace_module_suffix(
        pth_sup_model, f"{append_suffix}2.", f"{append_suffix}3."
    )
    pth_sup_model = replace_module_suffix(
        pth_sup_model, f"{append_suffix}1.", f"{append_suffix}2."
    )
    pth_sup_model = replace_module_suffix(
        pth_sup_model, f"{append_suffix}bn1.", f"{append_suffix}0.1."
    )
    pth_sup_model = replace_module_suffix(
        pth_sup_model, f"{append_suffix}conv1.", f"{append_suffix}0.0."
    )
    state = {"model_state_dict": pth_sup_model}
    logger.info(f"Converted model. Number of params: {len(pth_sup_model.keys())}")
    output_filename = f"converted_{os.path.basename(model_path)}"
    output_model_filepath = os.path.join(args.output_dir, output_filename)
    logger.info(f"Saving model: {output_model_filepath}")
    torch.save(state, output_model_filepath)
    logger.info("DONE!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch RN50 supervised model weights to VISSL"
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
    args = parser.parse_args()
    convert_and_save_model(args, append_suffix="_feature_blocks.")


if __name__ == "__main__":
    main()
