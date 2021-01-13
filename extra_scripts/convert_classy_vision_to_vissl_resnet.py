# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script is used to convert the ClassyVision ResNet models to VISSL/torchvision
compatible ResNet models
"""
import argparse
import logging
import sys

import torch


# create the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


_SKIP_LAYERS_IN_TRUNK = ["fc", "clf", "num_batches_tracked"]

# For more depths, add the block config here
BLOCK_CONFIG = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


def convert_classy_trunk_to_vissl_model(state_dict_trunk, depth):
    assert depth in BLOCK_CONFIG.keys(), f"depth {depth} conversion not supported"
    layers = BLOCK_CONFIG[depth]

    output_dict = {}
    for (k, val) in state_dict_trunk.items():
        if any(x in k for x in _SKIP_LAYERS_IN_TRUNK):
            continue
        k = k.replace("initial_block._module.0.", "_feature_blocks.conv1.")
        k = k.replace("initial_block._module.1.", "_feature_blocks.bn1.")
        for idx in range(len(layers)):
            num_blocks = layers[idx]
            for j in range(num_blocks):
                k = k.replace(
                    f"blocks.{idx}.block{idx}-{j}.",
                    f"_feature_blocks.layer{idx + 1}.{j}.",
                )
        k = k.replace(".convolutional_block.0.weight", ".conv1.weight")
        k = k.replace(".convolutional_block.3.weight", ".conv2.weight")
        k = k.replace(".convolutional_block.6.weight", ".conv3.weight")
        k = k.replace(".convolutional_block.1.", ".bn1.")
        k = k.replace(".convolutional_block.4.", ".bn2.")
        k = k.replace(".bn.", ".bn3.")
        output_dict[k] = val
    return output_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert SSL framework RN50 models to Classy Vision models"
    )
    parser.add_argument(
        "--depth", type=int, default=50, help="Depth of the ResNet model to convert"
    )
    parser.add_argument(
        "--input_model_file",
        type=str,
        default=None,
        help="Path to input model weights to be converted",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=None,
        help="Path to save converted RN-50 model",
    )
    args = parser.parse_args()

    # load the input model weights
    logger.info("Loading weights...")
    state_dict = torch.load(args.input_model_file)
    assert "classy_state_dict" in state_dict, "classy_state_dict not found"
    state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]

    converted_trunk = convert_classy_trunk_to_vissl_model(state_dict, args.depth)
    output_state_dict = {"model_state_dict": converted_trunk}
    logger.info("Saving converted weights to: {}".format(args.output_model))
    torch.save(output_state_dict, args.output_model)
    logger.info("Done!!")


if __name__ == "__main__":
    main()
