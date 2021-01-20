# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from enum import Enum
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from vissl.utils.activation_checkpointing import checkpoint_trunk
from vissl.utils.misc import is_apex_available


# Tuple of classes of BN layers.
_bn_cls = (nn.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm)


if is_apex_available():
    import apex

    try:
        # try importing the optimized version directly
        _bn_cls = _bn_cls + (apex.parallel.optimized_sync_batchnorm.SyncBatchNorm,)
    except AttributeError:
        _bn_cls = _bn_cls + (apex.parallel.SyncBatchNorm,)


def transform_model_input_data_type(model_input, model_config):
    """
    Default model input follow RGB format. Based the model input specified,
    change the type. Supported types: RGB, BGR, LAB
    """
    model_output = model_input
    # In case the model takes BGR input type, we convert the RGB to BGR
    if model_config.INPUT_TYPE == "bgr":
        model_output = model_input[:, [2, 1, 0], :, :]
    # In case of LAB image, we take only "L" channel as input. Split the data
    # along the channel dimension into [L, AB] and keep only L channel.
    if model_config.INPUT_TYPE == "lab":
        model_output = torch.split(model_input, [1, 2], dim=1)[0]
    return model_output


def is_feature_extractor_model(model_config):
    """
    If the model is a feature extractor model:
        - evaluation model is on
        - trunk is frozen
        - number of features specified for features extratction > 0
    """
    if (
        model_config.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON
        and model_config.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY
        and len(model_config.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP) > 0
    ):
        return True
    return False


def get_trunk_output_feature_names(model_config):
    """
    Get the feature names which we will use to associate the features witl.
    If Feature eval mode is set, we get feature names from
    config.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP.
    """
    feature_names = []
    if is_feature_extractor_model(model_config):
        feat_ops_map = model_config.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP
        feature_names = [item[0] for item in feat_ops_map]
    return feature_names


class Wrap(nn.Module):
    """
    Wrap a free function into a nn.Module.
    Can be useful to build a model block, and include activations or light tensor alterations
    """

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class SyncBNTypes(str, Enum):
    """
    Supported SyncBN types
    """

    apex = "apex"
    pytorch = "pytorch"


def convert_sync_bn(config, model):
    """
    Convert the BatchNorm layers in the model to the SyncBatchNorm layers.

    For SyncBatchNorm, we support two sources: Apex and PyTorch. The optimized
    SyncBN kernels provided by apex run faster.

    Args:
        config (AttrDict): configuration file
        model: Pytorch model whose BatchNorm layers should be converted to SyncBN
               layers.

    NOTE: Since SyncBatchNorm layer synchronize the BN stats across machines, using
          the syncBN layer can be slow. In order to speed up training while using
          syncBN, we recommend using process_groups which are very well supported
          for Apex.
          To set the process groups, set SYNC_BN_CONFIG.GROUP_SIZE following below:
          1) if group_size=-1 -> use the VISSL default setting. We synchronize within a
             machine and hence will set group_size=num_gpus per node. This gives the best
             speedup.
          2) if group_size>0 -> will set group_size=value set by user.
          3) if group_size=0 -> no groups are created and process_group=None. This means
             global sync is done.
    """
    sync_bn_config = config.MODEL.SYNC_BN_CONFIG

    def get_group_size():

        world_size = config.DISTRIBUTED.NUM_PROC_PER_NODE * config.DISTRIBUTED.NUM_NODES
        if sync_bn_config["GROUP_SIZE"] > 0:
            # if the user specifies group_size to create, we use that.
            # we also make sure additionally that the group size doesn't exceed
            # the world_size. This is beneficial to handle especially in case
            # of 1 node training where num_gpu <= 8
            group_size = min(world_size, sync_bn_config["GROUP_SIZE"])
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
            # import os
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
    """
    Flatten module attached in the model. It basically flattens the input tensor.
    """

    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        """
        flatten the input feat
        """
        return torch.flatten(feat, start_dim=self.dim)

    def flops(self, x):
        """
        number of floating point operations performed. 0 for this module.
        """
        return 0


class Identity(nn.Module):
    """
    A helper module that outputs the input as is
    """

    def __init__(self, args=None):
        super().__init__()

    def forward(self, x):
        """
        Return the input as the output
        """
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
    """
    Types of Norms supported in ResNe(X)t trainings. can be easily set and modified
    from the config file.
    """

    BatchNorm = "BatchNorm"
    LayerNorm = "LayerNorm"


def _get_norm(layer_name):
    """
    return the normalization layer to use in the model based on the layer name
    """
    return {
        RESNET_NORM_LAYER.BatchNorm: nn.BatchNorm2d,
        RESNET_NORM_LAYER.LayerNorm: LayerNorm2d,
    }[layer_name]


def parse_out_keys_arg(
    out_feat_keys: List[str], all_feat_names: List[str]
) -> Tuple[List[str], int]:
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


def get_trunk_forward_outputs_module_list(
    feat: torch.Tensor,
    out_feat_keys: List[str],
    feature_blocks: nn.ModuleList,
    all_feat_names: List[str] = None,
) -> List[torch.Tensor]:
    """
    Args:
        feat: model input.
        out_feat_keys: a list/tuple with the feature names of the features that
            the function should return. By default the last feature of the network
            is returned.
        feature_blocks: list of feature blocks in the model
        feature_mapping: name of the layers in the model

    Returns:
        out_feats: a list with the asked output features placed in the same order as in
        `out_feat_keys`.
    """
    out_feat_keys, max_out_feat = parse_out_keys_arg(out_feat_keys, all_feat_names)
    out_feats = [None] * len(out_feat_keys)
    for f in range(max_out_feat + 1):
        feat = feature_blocks[f](feat)
        key = all_feat_names[f]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat
    return out_feats


def get_trunk_forward_outputs(
    feat: torch.Tensor,
    out_feat_keys: List[str],
    feature_blocks: nn.ModuleDict,
    feature_mapping: Dict[str, str] = None,
    use_checkpointing: bool = True,
    checkpointing_splits: int = 2,
) -> List[torch.Tensor]:
    """
    Args:
        feat: model input.
        out_feat_keys: a list/tuple with the feature names of the features that
            the function should return. By default the last feature of the network
            is returned.
        feature_blocks: ModuleDict containing feature blocks in the model
        feature_mapping: an optional correspondence table in between the requested
            feature names and the model's.

    Returns:
        out_feats: a list with the asked output features placed in the same order as in
        `out_feat_keys`.
    """

    # Sanitize inputs
    if feature_mapping is not None:
        out_feat_keys = [feature_mapping[f] for f in out_feat_keys]

    out_feat_keys, max_out_feat = parse_out_keys_arg(
        out_feat_keys, list(feature_blocks.keys())
    )

    # Forward pass over the trunk
    unique_out_feats = {}
    unique_out_feat_keys = list(set(out_feat_keys))

    # FIXME: Ideally this should only be done once at construction time
    if use_checkpointing:
        feature_blocks = checkpoint_trunk(
            feature_blocks, unique_out_feat_keys, checkpointing_splits
        )

        # If feat is the first input to the network, it doesn't have requires_grad,
        # which will make checkpoint's backward function not being called. So we need
        # to set it to true here.
        feat.requires_grad = True

    # Go through the blocks, and save the features as we go
    # NOTE: we are not doing several forward passes but instead just checking
    # whether the feature should is requested to be returned.
    for i, (feature_name, feature_block) in enumerate(feature_blocks.items()):
        # The last chunk has to be non-volatile
        if use_checkpointing and i < len(feature_blocks) - 1:
            # Un-freeze the running stats in any BN layer
            for m in filter(lambda x: isinstance(x, _bn_cls), feature_block.modules()):
                m.track_running_stats = m.training

            feat = checkpoint(feature_block, feat)

            # Freeze the running stats in any BN layer
            # the checkpointing process will have to do another FW pass
            for m in filter(lambda x: isinstance(x, _bn_cls), feature_block.modules()):
                m.track_running_stats = False
        else:
            feat = feature_block(feat)

        # This feature is requested, store. If the same feature is requested several
        # times, we return the feature several times.
        if feature_name in unique_out_feat_keys:
            unique_out_feats[feature_name] = feat

        # Early exit if all the features have been collected
        if i == max_out_feat and not use_checkpointing:
            break

    # now return the features as requested by the user. If there are no duplicate keys,
    # return as is.
    if len(unique_out_feat_keys) == len(out_feat_keys):
        return list(unique_out_feats.values())

    output_feats = []
    for key_name in out_feat_keys:
        output_feats.append(unique_out_feats[key_name])

    return output_feats
