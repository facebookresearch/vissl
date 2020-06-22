#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.abs

import logging
import os
from enum import Enum, auto

import torch
import torch.nn as nn
from vissl.utils.misc import is_apex_available


if is_apex_available():
    import apex


class SyncBNTypes(str, Enum):
    apex = "apex"
    pytorch = "pytorch"


def convert_sync_bn(config, model):
    sync_bn_config = config.MODEL.SYNC_BN_CONFIG

    def get_group_size():
        if sync_bn_config["GROUP_SIZE"] > 0:
            # if the user specifies group_size to create, we use that.
            group_size = sync_bn_config["GROUP_SIZE"]
        elif sync_bn_config["GROUP_SIZE"] == 0:
            # group_size=0 is considered as world_size and no process group is created.
            group_size = None
        else:
            # by default, we set it to number of gpus in a node. Within gpu, the
            # interconnect is fast and syncBN is cheap.
            group_size = config.DISTRIBUTED.NUM_PROC_PER_NODE
        logging.info(f"Using SyncBN group size: {group_size}")
        return group_size

    def to_apex_syncbn(group_size):
        logging.info("Converting BN layers to Apex SyncBN")
        if group_size is None:
            process_group = None
            logging.info("Not creating process_group for Apex SyncBN...")
        else:
            process_group = apex.parallel.create_syncbn_process_group(
                group_size=group_size
            )
        return apex.parallel.convert_syncbn_model(model, process_group=process_group)

    def to_pytorch_syncbn(group_size):
        logging.info("Converting BN layers to PyTorch SyncBN")
        if group_size is None:
            process_group = None
            logging.info("Not creating process_group for PyTorch SyncBN...")
        else:
            logging.warning(
                "Process groups not supported with PyTorch SyncBN currently. "
                "Traning will be slow. Please consider installing Apex for SyncBN."
            )
            process_group = None
            # TODO (prigoyal): process groups don't work well with pytorch.
            # num_gpus_per_node = config.DISTRIBUTED.NUM_PROC_PER_NODE
            # node_id = int(os.environ["RANK"]) // num_gpus_per_node
            # assert (
            #     group_size == num_gpus_per_node
            # ), "Use group_size=num_gpus per node as interconnect is cheap in a machine"
            # process_ids = list(
            #     range(
            #         node_id * num_gpus_per_node,
            #         (node_id * num_gpus_per_node) + group_size,
            #     )
            # )
            # logging.info(f"PyTorch SyncBN Node: {node_id} process_ids: {process_ids}")
            # process_group = torch.distributed.new_group(process_ids)
        return nn.SyncBatchNorm.convert_sync_batchnorm(
            model, process_group=process_group
        )

    group_size = get_group_size()
    # Apply the correct transform, make sure that any other setting raises an error
    return {SyncBNTypes.apex: to_apex_syncbn, SyncBNTypes.pytorch: to_pytorch_syncbn}[
        sync_bn_config["SYNC_BN_TYPE"]
    ](group_size)


class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)

    def flops(self, x):
        return 0


class Identity(nn.Module):
    def __init__(self, args=None):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm2d(nn.GroupNorm):
    """
    Use GroupNorm to construct LayerNorm as pytorch LayerNorm2d requires
    specifying input_shape explicitly which is inconvenient. Set num_groups=1 to
    convert GroupNorm to LayerNorm.
    """

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2d, self).__init__(
            num_groups=1, num_channels=num_channels, eps=eps, affine=affine
        )


class RESNET_NORM_LAYER(str, Enum):
    BatchNorm = auto()
    LayerNorm = auto()


def _get_norm(layer_name):
    if RESNET_NORM_LAYER[layer_name] == RESNET_NORM_LAYER.BatchNorm:
        norm_layer = nn.BatchNorm2d
    elif RESNET_NORM_LAYER[layer_name].name == RESNET_NORM_LAYER.LayerNorm:
        norm_layer = LayerNorm2d
    return norm_layer


def parse_out_keys_arg(out_feat_keys, all_feat_names):
    """
    Checks if all out_feature_keys are mapped to a layer in the model.
    Returns the last layer to forward pass through for efficiency.
    Allow duplicate features also to be evaluated.
    Adapted from (https://github.com/gidariss/FeatureLearningRotNet).
    """

    # By default return the features of the last layer / module.
    if out_feat_keys is None or (len(out_feat_keys) == 0):
        out_feat_keys = [all_feat_names[-1]]

    if len(out_feat_keys) == 0:
        raise ValueError("Empty list of output feature keys.")
    for _, key in enumerate(out_feat_keys):
        if key not in all_feat_names:
            raise ValueError(
                f"Feature with name {key} does not exist. "
                f"Existing features: {all_feat_names}."
            )

    # Find the highest output feature in `out_feat_keys
    max_out_feat = max(all_feat_names.index(key) for key in out_feat_keys)

    return out_feat_keys, max_out_feat


def get_trunk_forward_outputs(feat, out_feat_keys, feature_blocks, all_feat_names):
    """
    Args:
        feat: model input.
        out_feat_keys: a list/tuple with the feature names of the features that
            the function should return. By default the last feature of the network
            is returned.
        feature_blocks: list of feature blocks in the model
        all_feat_names: list of different feature layers in the model

    Return:
        out_feats: If multiple output features were asked then `out_feats` is a
        list with the asked output features placed in the same order as in
        `out_feat_keys`. If a single output feature was asked then `out_feats`
        is that output feature (and not a list).
    """
    out_feat_keys, max_out_feat = parse_out_keys_arg(out_feat_keys, all_feat_names)
    out_feats = [None] * len(out_feat_keys)
    for f in range(max_out_feat + 1):
        feat = feature_blocks[f](feat)
        key = all_feat_names[f]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat
    return out_feats
