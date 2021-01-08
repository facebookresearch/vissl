# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script is used to convert the SSL Framework models to the Detectron2 compatible
models.
"""
import argparse
import logging
import re
import sys
from collections import OrderedDict

import numpy as np
import torch


# create the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# we skip the layers that belong to model head. We only convert the model trunk.
_SKIP_LAYERS = ["num_batches_tracked", "fc1", "fc2"]


def convert_to_detectron2_names(layer_keys):
    output_keys = []
    for k in layer_keys:
        k = k.replace("_feature_blocks.conv1.", "stem.conv1.")
        k = k.replace("_feature_blocks.bn1.", "stem.conv1.norm.")
        k = k.replace("_feature_blocks.layer1.", "res2.")
        k = k.replace("_feature_blocks.layer2.", "res3.")
        k = k.replace("_feature_blocks.layer3.", "res4.")
        k = k.replace("_feature_blocks.layer4.", "res5.")

        k = k.replace(".downsample.0.", ".shortcut.")
        k = k.replace(".downsample.1.", ".shortcut.norm.")
        k = k.replace(".bn1.", ".conv1.bn.")
        k = k.replace(".bn2.", ".conv2.bn.")
        k = k.replace(".bn3.", ".conv3.bn.")
        k = re.sub("bn\\.bias$", "norm.bias", k)
        k = re.sub("bn\\.weight$", "norm.weight", k)
        k = re.sub("bn\\.running_mean$", "norm.running_mean", k)
        k = re.sub("bn\\.running_var$", "norm.running_var", k)
        output_keys.append(k)

    assert len(output_keys) == len(set(output_keys)), "Error in converting layer names"
    return output_keys


def _rename_weights_to_d2(weights, weights_type):
    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())

    # basic layer mapping to detectron names
    layer_keys = convert_to_detectron2_names(layer_keys)
    key_map = {k: v for k, v in zip(original_keys, layer_keys)}

    logger.info("Remapping weights....")
    new_weights = OrderedDict()
    for k in original_keys:
        if any(x in k for x in _SKIP_LAYERS):
            continue
        if weights_type == "torch":
            v = np.array(weights[k].detach().cpu())
        else:
            v = np.array(weights[k])
        w = torch.from_numpy(v)
        logger.info(f"original name: {k} \t\t mapped name: {key_map[k]}")
        new_weights[key_map[k]] = w
    logger.info("Number of params: {}".format(len(new_weights)))
    return new_weights


def main():
    parser = argparse.ArgumentParser(
        description="Convert SSL framework RN50 model to Detectron2"
    )
    parser.add_argument(
        "--input_model_file",
        type=str,
        default=None,
        help="Path to input model weights to be converted",
    )
    parser.add_argument(
        "--output_model", type=str, default=None, help="Path to save torch RN-50 model"
    )
    parser.add_argument(
        "--state_dict_key_name",
        type=str,
        default="model_state_dict",
        help="Pytorch model state_dict key name",
    )
    parser.add_argument(
        "--weights_type", type=str, required=True, default="numpy", help="numpy | torch"
    )
    args = parser.parse_args()

    # load the input model weights
    logger.info("Loading weights...")
    vissl_state_dict = torch.load(args.input_model_file)
    assert (
        args.state_dict_key_name in vissl_state_dict
    ), f"{args.state_dict_key_name} not found"
    vissl_state_dict = vissl_state_dict[args.state_dict_key_name]
    if args.state_dict_key_name == "classy_state_dict":
        vissl_state_dict = vissl_state_dict["base_model"]["model"]["trunk"]

    renamed_state_dict = _rename_weights_to_d2(vissl_state_dict, args.weights_type)
    state = {"model": renamed_state_dict, "matching_heuristics": True}
    logger.info("Saving converted weights to: {}".format(args.output_model))
    torch.save(state, args.output_model)
    logger.info("Done!!")


if __name__ == "__main__":
    main()
