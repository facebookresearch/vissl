# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert the ResNet-50 models from ICCV'19 paper https://arxiv.org/abs/1905.01235
to PyTorch TorchVision compatible models. We can directly use these models
for benchmarking.

Code credits:
https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/extra_scripts/pickle_caffe2_to_pytorch.py # NOQA
"""
import argparse
import logging
import pickle
import re
import sys
from collections import OrderedDict

import torch
from iopath.common.file_io import g_pathmgr


# create the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


_C2_STAGE_NAMES = {"R-50": ["1.2", "2.3", "3.5", "4.2"]}


def remove_jigsaw_names(data):
    output_blobs, count = {}, 0
    logger.info("Correcting jigsaw model...")
    remove_suffixes = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    for item in sorted(data.keys()):
        if "s0" in item:
            out_name = re.sub("_s[0-9]_", "_", item)
            logger.info("input_name: {} out_name: {}".format(item, out_name))
            output_blobs[out_name] = data[item]
        elif any(x in item for x in remove_suffixes):
            count += 1
            logger.info("Ignoring: {}".format(item))
        else:
            logger.info("adding: {}".format(item))
            output_blobs[item] = data[item]

    logger.info("Original #blobs: {}".format(len(data.keys())))
    logger.info("Output #blobs: {}".format(len(output_blobs.keys())))
    logger.info("Removed #blobs: {}".format(count))
    return output_blobs


def _rename_basic_resnet_weights(layer_keys):
    layer_keys = [k.replace("_", ".") for k in layer_keys]
    layer_keys = [k.replace(".w", ".weight") for k in layer_keys]
    layer_keys = [k.replace(".bn", "_bn") for k in layer_keys]
    layer_keys = [k.replace(".b", ".bias") for k in layer_keys]
    layer_keys = [k.replace("_bn.s", "_bn.scale") for k in layer_keys]
    layer_keys = [k.replace(".biasranch", ".branch") for k in layer_keys]
    layer_keys = [k.replace("res.conv1_", "conv1_") for k in layer_keys]

    # Affine-Channel -> BatchNorm enaming
    layer_keys = [k.replace("_bn.scale", "_bn.weight") for k in layer_keys]
    layer_keys = [k.replace("_bn.rm", "_bn.running_mean") for k in layer_keys]
    layer_keys = [k.replace("_bn.riv", "_bn.running_var") for k in layer_keys]

    # Make torchvision-compatible
    layer_keys = [k.replace("conv1_bn.", "bn1.") for k in layer_keys]

    layer_keys = [k.replace("res2.", "layer1.") for k in layer_keys]
    layer_keys = [k.replace("res3.", "layer2.") for k in layer_keys]
    layer_keys = [k.replace("res4.", "layer3.") for k in layer_keys]
    layer_keys = [k.replace("res5.", "layer4.") for k in layer_keys]

    layer_keys = [k.replace(".branch2a.", ".conv1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2a_bn.", ".bn1.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b.", ".conv2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2b_bn.", ".bn2.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c.", ".conv3.") for k in layer_keys]
    layer_keys = [k.replace(".branch2c_bn.", ".bn3.") for k in layer_keys]

    layer_keys = [k.replace(".branch1.", ".downsample.0.") for k in layer_keys]
    layer_keys = [k.replace(".branch1_bn.", ".downsample.1.") for k in layer_keys]

    return layer_keys


def _rename_weights_for_resnet(weights, stage_names):
    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())

    # for X-101, rename output to fc1000 to avoid conflicts afterwards
    layer_keys = [k if k != "pred_b" else "fc1000_b" for k in layer_keys]
    layer_keys = [k if k != "pred_w" else "fc1000_w" for k in layer_keys]

    # performs basic renaming: _ -> . , etc
    layer_keys = _rename_basic_resnet_weights(layer_keys)
    key_map = {k: v for k, v in zip(original_keys, layer_keys)}

    logger.info("Remapping C2 weights")
    max_c2_key_size = max(len(k) for k in original_keys if "_momentum" not in k)

    new_weights = OrderedDict()
    for k in original_keys:
        v = weights[k]
        if "_momentum" in k:
            continue
        if "pred" in k:
            continue
        if k == "lr":
            continue
        if k == "model_iter":
            continue
        w = torch.from_numpy(v)
        logger.info(
            "C2 name: {: <{}} mapped name: {}".format(k, max_c2_key_size, key_map[k])
        )
        new_weights[key_map[k]] = w
    logger.info("Number of blobs: {}".format(len(new_weights)))
    return new_weights


def _load_c2_pickled_weights(file_path):
    with g_pathmgr.open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    if "blobs" in data:
        weights = data["blobs"]
    else:
        weights = data
    return weights


def convert_bgr2rgb(state_dict):
    w = state_dict["conv1_w"]  # (64, 3, 7, 7)
    assert w.shape == (64, 3, 7, 7), "Error in converting bgr2rgb"
    w = w[:, ::-1, :, :]
    state_dict["conv1_w"] = w.copy()
    logger.info("BGR ===> RGB for conv1_w.")
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert C2 model to Pytorch")
    parser.add_argument(
        "--c2_model", type=str, default=None, help="Path to c2 RN-50 model"
    )
    parser.add_argument(
        "--output_model", type=str, default=None, help="Path to save torch RN-50 model"
    )
    parser.add_argument(
        "--bgr2rgb",
        dest="bgr2rgb",
        default=False,
        help="Revert bgr (openCV) order to rgb (PIL) order",
    )
    parser.add_argument(
        "--jigsaw", type=bool, default=False, help="Whether jigsaw model or not"
    )
    parser.add_argument("--arch", type=str, default="R-50", help="R-50")
    args = parser.parse_args()

    stages = _C2_STAGE_NAMES[args.arch]

    # load the caffe2 model weights
    state_dict = _load_c2_pickled_weights(args.c2_model)

    # for the pretext model from jigsaw, special processing
    if args.jigsaw:
        state_dict = remove_jigsaw_names(state_dict)

    # depending on the image reading library, we convert the weights to be
    # compatible order. The default order of caffe2 weights is BGR (openCV).
    if args.bgr2rgb:
        state_dict = convert_bgr2rgb(state_dict)

    state_dict = _rename_weights_for_resnet(state_dict, stages)
    logger.info("Saving converted weights to: {}".format(args.output_model))
    torch.save(state_dict, args.output_model)
    logger.info("Done!!")


if __name__ == "__main__":
    main()
