# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
The script does following 3 kinds of conversions:
- Convert the AlexNet models from ICCV'19 paper https://arxiv.org/abs/1905.01235
trained in Caffe2 for self-supervised approaches Jigsaw, Colorization and supervised
training on ImageNet-1k, Places-205 to PyTorch weights compatible with VISSL.
- Convert the PyTorch model for AlexNet DeepCluster available at
https://github.com/facebookresearch/deepcluster to weights compatible with VISSL.
- Convert the PyTorch model for AlexNet RotNet available at
https://github.com/gidariss/FeatureLearningRotNet to weights compatible with VISSL.

You can easily extend this script to convert your models.
"""
import argparse
import logging
import pickle
import sys
from collections import OrderedDict

import numpy as np
import torch
from fvcore.common.file_io import PathManager


# create the logger
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# we skip the layers that belong to model head. We only convert the model trunk.
_SKIP_LAYERS = ["_momentum", "sobel", "classifier", "top_layer", "pred", "fc", "data"]


def remove_jigsaw_names(data):
    output_blobs, count = {}, 0
    logger.info("Correcting jigsaw model...")
    remove_suffixes = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
    for item in sorted(data.keys()):
        if "s0" in item:
            out_name = item.replace("_s0_", "_")
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


def _load_c2_pickled_weights(file_path):
    with PathManager.open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    if "blobs" in data:
        weights = data["blobs"]
    else:
        weights = data
    return weights


def _load_c2_weights(file_path):
    if file_path.endswith("pkl"):
        weights = _load_c2_pickled_weights(file_path)
    elif file_path.endswith("npy"):
        with PathManager.open(file_path, "rb") as fopen:
            weights = np.load(fopen, allow_pickle=True, encoding="latin1")[()]
    return weights


def load_weights(file_path, weights_type, model_name):
    weights = None
    if weights_type == "caffe2":
        weights = _load_c2_weights(file_path)
    elif weights_type == "torch":
        weights = torch.load(file_path)
        if model_name == "deepcluster":
            weights = weights["state_dict"]
        elif model_name == "rotnet":
            weights = weights["network"]
        else:
            logger.critical(f"Unknown {weights_type} model: {model_name}")
    else:
        logger.error("Only caffe2 and torch weights are supported")
    return weights


def convert_bgr2rgb(state_dict):
    w = state_dict["conv1_w"]
    w = w[:, ::-1, :, :]
    state_dict["conv1_w"] = w.copy()
    logger.info("BGR ===> RGB for conv1_w.")
    return state_dict


def _rename_basic_caffe2_alexnet_weights(layer_keys):
    output_keys = []
    for k in layer_keys:
        k = k.replace("norm", "_bn")
        k = k.replace("_bn_b", ".1.bias")
        k = k.replace("_bn_s", ".1.weight")
        k = k.replace("_bn_rm", ".1.running_mean")
        k = k.replace("_bn_riv", ".1.running_var")
        k = k.replace("_b", ".0.bias")
        k = k.replace("_w", ".0.weight")
        k = k.replace("conv1.", "0.")
        k = k.replace("conv2.", "2.")
        k = k.replace("conv3.", "4.")
        k = k.replace("conv4.", "5.")
        k = k.replace("conv5.", "6.")
        k = f"_feature_blocks.{k}"
        output_keys.append(k)
    return output_keys


def _rename_basic_alexnet_deepcluster_weights(layer_keys):
    output_keys = []
    for k in layer_keys:
        k = k.replace("features.module.0.", "0.0.")
        k = k.replace("features.module.1.", "0.1.")
        k = k.replace("features.module.4.", "2.0.")
        k = k.replace("features.module.5.", "2.1.")
        k = k.replace("features.module.8.", "4.0.")
        k = k.replace("features.module.9.", "4.1.")
        k = k.replace("features.module.11.", "5.0.")
        k = k.replace("features.module.12.", "5.1.")
        k = k.replace("features.module.14.", "6.0.")
        k = k.replace("features.module.15.", "6.1.")
        k = f"_feature_blocks.{k}"
        output_keys.append(k)
    return output_keys


def _rename_weights_for_alexnet(weights, weights_type, model_name):
    original_keys = sorted(weights.keys())
    layer_keys = sorted(weights.keys())

    # performs basic renaming: _ -> . , etc
    if weights_type == "caffe2":
        layer_keys = _rename_basic_caffe2_alexnet_weights(layer_keys)
    elif weights_type == "torch" and model_name == "deepcluster":
        layer_keys = _rename_basic_alexnet_deepcluster_weights(layer_keys)
    elif weights_type == "torch" and model_name == "rotnet":
        logger.info("Found rotnet torch weights compatible with vissl")
    key_map = {k: v for k, v in zip(original_keys, layer_keys)}

    logger.info("Remapping weights....")
    new_weights = OrderedDict()
    for k in original_keys:
        if weights_type == "torch":
            v = np.array(weights[k].cpu())
        else:
            v = np.array(weights[k])
        if any(x in k for x in _SKIP_LAYERS):
            continue
        if k == "lr":
            continue
        if k == "model_iter":
            continue
        w = torch.from_numpy(v)
        logger.info(f"original name: {k} \t\t mapped name: { key_map[k]}")
        new_weights[key_map[k]] = w
    logger.info("Number of params: {}".format(len(new_weights)))
    return new_weights


def main():
    parser = argparse.ArgumentParser(description="Convert AlexNet model to Pytorch")
    parser.add_argument(
        "--input_model_weights",
        type=str,
        default=None,
        help="Path to input model weights to be converted",
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
        "--model_name",
        type=str,
        required=True,
        help="jigsaw | colorization | supervised | rotnet | deepcluster",
    )  # NOQA
    parser.add_argument(
        "--weights_type", type=str, required=True, help="caffe2 | torch"
    )
    args = parser.parse_args()

    # load the input model weights
    logger.info("Loading weights...")
    state_dict = load_weights(
        args.input_model_weights, args.weights_type, args.model_name
    )

    # for the pretext model from jigsaw, special processing
    if args.model_name == "jigsaw":
        state_dict = remove_jigsaw_names(state_dict)

    # depending on the image reading library, we convert the weights to be
    # compatible order. The default order of caffe2 weights is BGR (openCV).
    if args.bgr2rgb:
        state_dict = convert_bgr2rgb(state_dict)

    state_dict = _rename_weights_for_alexnet(
        state_dict, args.weights_type, args.model_name
    )
    state = {"model_state_dict": state_dict}
    logger.info("Saving converted weights to: {}".format(args.output_model))
    torch.save(state, args.output_model)
    logger.info("Done!!")


if __name__ == "__main__":
    main()
