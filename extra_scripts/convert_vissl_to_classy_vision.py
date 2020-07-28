# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script is used to convert the SSL Framework models to the Detectron2 compatible
models.
"""
import argparse
import logging
import sys

import torch


# create the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


_SKIP_LAYERS_IN_TRUNK = ["fc", "clf"]

# For more depths, add the block config here
BLOCK_CONFIG = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}


def convert_trunk_to_classy_model(state_dict_trunk, depth):
    assert depth in BLOCK_CONFIG.keys(), f"depth {depth} conversion not supported"
    layers = BLOCK_CONFIG[depth]

    output_dict = {}
    for (k, val) in state_dict_trunk.items():
        if any(x in k for x in _SKIP_LAYERS_IN_TRUNK):
            continue
        k = k.replace("_feature_blocks.conv1.", "initial_block._module.0.")
        k = k.replace("_feature_blocks.bn1.", "initial_block._module.1.")
        for idx in range(len(layers)):
            num_blocks = layers[idx]
            for j in range(num_blocks):
                k = k.replace(
                    f"_feature_blocks.layer{idx + 1}.{j}.",
                    f"blocks.{idx}.block{idx}-{j}.",
                )
        k = k.replace(".conv1.weight", ".convolutional_block.0.weight")
        k = k.replace(".conv2.weight", ".convolutional_block.3.weight")
        k = k.replace(".conv3.weight", ".convolutional_block.6.weight")
        k = k.replace(".bn1.", ".convolutional_block.1.")
        k = k.replace(".bn2.", ".convolutional_block.4.")
        k = k.replace(".bn3.", ".bn.")
        output_dict[k] = val
    return output_dict


def convert_heads_to_classy_model(
    state_dict, out_prefix, num_fc_layers, use_bn_head=False, use_bias_head_fc=True
):
    """
    Convert an MLP head of 2 kinds:
    Type 1: FC only
    Type 2: FC -> RELU -> FC
    """
    logger.info("Converting head...")
    converted_dict = {"block3-2": {"default_head": {}}}
    if num_fc_layers > 1:
        out_dict = {}
        for idx in range(num_fc_layers - 1):
            # convert the linear layer
            local_prefix = f"mlp.layers.{idx}"
            out_dict[f"{local_prefix}.fc.weight"] = state_dict[f"{idx}.clf.0.weight"]
            if use_bias_head_fc:
                # linear layer bias can be optional
                out_dict[f"{local_prefix}.fc.bias"] = state_dict[f"{idx}.clf.0.bias"]

            # convert BN if we have BN in the head
            if use_bn_head:
                bn = f"{local_prefix}.batch_norm"
                out_dict[f"{bn}.weight"] = state_dict[f"{idx}.clf.1.weight"]
                out_dict[f"{bn}.bias"] = state_dict[f"{idx}.clf.1.bias"]
                out_dict[f"{bn}.running_mean"] = state_dict[f"{idx}.clf.1.running_mean"]
                out_dict[f"{bn}.running_var"] = state_dict[f"{idx}.clf.1.running_var"]

        # for the final output layer, the name is embedding_fc.weight/bias
        out_dict[f"{out_prefix}_fc.weight"] = state_dict[f"{idx + 1}.clf.0.weight"]
        if use_bias_head_fc:
            out_dict[f"{out_prefix}_fc.bias"] = state_dict[f"{idx + 1}.clf.0.bias"]
        converted_dict["block3-2"]["default_head"] = out_dict
    else:
        # for converting NPID models, there's only one projection FC
        converted_dict = {
            "block3-2": {
                "default_head": {
                    f"{out_prefix}_fc.weight": state_dict["0.clf.0.weight"],
                    f"{out_prefix}_fc.bias": state_dict["0.clf.0.bias"],
                }
            }
        }
    return converted_dict


def main():
    parser = argparse.ArgumentParser(
        description="Convert SSL framework RN50 models to Classy Vision models"
    )
    parser.add_argument(
        "--depth", type=int, default=50, help="Depth of the ResNet model to convert"
    )
    parser.add_argument(
        "--num_fc_layers",
        type=int,
        default=1,
        help="Number of linear layers in the head",
    )
    parser.add_argument(
        "--include_heads",
        default=False,
        action="store_true",
        help="Whether to convert the head as well or not",
    )
    parser.add_argument(
        "--use_bn_head",
        default=False,
        action="store_true",
        help="Whether BN is in the head or not",
    )
    parser.add_argument(
        "--use_bias_head_fc",
        default=False,
        action="store_true",
        help="Whether FC layers in head have bias param or not",
    )
    parser.add_argument(
        "--output_head_prefix",
        type=str,
        default="embedding",
        help="The [output_prefix]_fc.weight for the heads",
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
    parser.add_argument(
        "--state_dict_key_name",
        type=str,
        default="classy_state_dict",
        help="Pytorch model state_dict key name",
    )
    args = parser.parse_args()

    # load the input model weights
    logger.info("Loading weights...")
    state_dict = torch.load(args.input_model_file)
    assert (
        args.state_dict_key_name in state_dict
    ), f"{args.state_dict_key_name} not found"
    state_dict = state_dict[args.state_dict_key_name]

    if args.state_dict_key_name == "classy_state_dict":
        state_dict_head = state_dict["base_model"]["model"]["heads"]
        state_dict = state_dict["base_model"]["model"]["trunk"]
    else:
        assert not args.include_heads, "Can't convert heads"

    converted_trunk = convert_trunk_to_classy_model(state_dict, args.depth)
    if args.include_heads:
        converted_heads = convert_heads_to_classy_model(
            state_dict_head,
            args.output_head_prefix,
            args.num_fc_layers,
            args.use_bn_head,
            args.use_bias_head_fc,
        )
        output_state_dict = {
            "classy_state_dict": {
                "base_model": {
                    "model": {"trunk": converted_trunk, "heads": converted_heads}
                }
            }
        }
    else:
        output_state_dict = {
            "classy_state_dict": {"base_model": {"model": {"trunk": converted_trunk}}}
        }
    logger.info("Saving converted weights to: {}".format(args.output_model))
    torch.save(output_state_dict, args.output_model)
    logger.info("Done!!")


if __name__ == "__main__":
    main()
