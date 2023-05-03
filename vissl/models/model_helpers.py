# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import warnings
from enum import Enum
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _ntuple
from torch.utils.checkpoint import checkpoint
from vissl.data.collators.collator_helper import MultiDimensionalTensor
from vissl.utils.activation_checkpointing import checkpoint_trunk
from vissl.utils.env import get_machine_local_and_dist_rank
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


def transform_model_input_data_type(model_input, input_type: str):
    """
    Default model input follow RGB format. Based the model input specified,
    change the type. Supported types: RGB, BGR, LAB
    """
    model_output = model_input
    # In case the model takes BGR input type, we convert the RGB to BGR
    if input_type == "bgr":
        model_output = model_input[:, [2, 1, 0], :, :]
    # In case of LAB image, we take only "L" channel as input. Split the data
    # along the channel dimension into [L, AB] and keep only L channel.
    if input_type == "lab":
        model_output = torch.split(model_input, [1, 2], dim=1)[0]
    return model_output


def model_output_has_nan(model_output) -> bool:
    """
    Model output can be:
    - a tensor
    - list of tensors
    - list of list of tensors
    """
    from vissl.losses.cross_entropy_multiple_output_single_target import EnsembleOutput

    if isinstance(model_output, list):
        return any(model_output_has_nan(x) for x in model_output)
    elif isinstance(model_output, EnsembleOutput):
        return not torch.isfinite(model_output.outputs).all()
    else:
        return not torch.isfinite(model_output).all()


def is_feature_extractor_model(model_config):
    """
    If the model is a feature extractor model:
        - evaluation model is on
        - trunk is frozen
        - number of features specified for features extraction > 0
    """
    return (
        model_config.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON
        and (
            model_config.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY
            or model_config.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_AND_HEAD
        )
        and len(model_config.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP) > 0
    )


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


def get_no_ddp_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


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


def split_world_in_process_groups(world_size: int, group_size: int) -> List[List[int]]:
    """
    Split the process ids of the worlds (from 0 to world_size-1) into chunks
    of size bounded by group_size.

    Examples:

        > split_world_in_process_groups(world_size=9, group_size=3)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        > split_world_in_process_groups(world_size=9, group_size=4)
        [[0, 1, 2, 3], [4, 5, 6, 7], [8]]

        > split_world_in_process_groups(world_size=15, group_size=4)
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14]]

    """
    all_groups = []
    all_ids = list(reversed(range(world_size)))
    while all_ids:
        all_groups.append(all_ids[-group_size:][::-1])
        all_ids = all_ids[:-group_size]
    return all_groups


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
            process_group_ids = split_world_in_process_groups(
                world_size=config.DISTRIBUTED.NUM_PROC_PER_NODE
                * config.DISTRIBUTED.NUM_NODES,
                group_size=group_size,
            )
            process_groups = [dist.new_group(pids) for pids in process_group_ids]
            _, dist_rank = get_machine_local_and_dist_rank()
            process_group = process_groups[dist_rank // group_size]
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
    GroupNorm = "GroupNorm"


def _get_norm(trunk_config):
    """
    return the normalization layer to use in the model based on the layer name
    """
    layer_name = trunk_config.NORM
    n_groups = trunk_config.GROUPNORM_GROUPS

    def group_norm(num_channels):
        return nn.GroupNorm(num_groups=n_groups, num_channels=num_channels)

    return {
        RESNET_NORM_LAYER.BatchNorm: nn.BatchNorm2d,
        RESNET_NORM_LAYER.LayerNorm: LayerNorm2d,
        RESNET_NORM_LAYER.GroupNorm: group_norm,
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


def rearrange(x: torch.Tensor, pattern: str) -> torch.Tensor:
    """
    Rearranges a tensor by permuting its inputs based on a pattern
    provided as input

    Example:

        rearrange(torch.randn(size=(2, 3, 4, 5, 6)), 'n d h w c -> n c d h w').shape
        > torch.Size([2, 6, 3, 4, 5])
    """
    before, after = pattern.split("->")
    before = before.strip().split(" ")
    after = after.strip().split(" ")
    after = [before.index(a) for a in after]
    assert len(after) == len(before)
    return x.permute(after)


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


def get_tunk_forward_interpolated_outputs(
    input_type: str,  # bgr or rgb or lab
    interpolate_out_feat_key_name: str,
    remove_padding_before_feat_key_name: str,
    feat: MultiDimensionalTensor,
    feature_blocks: nn.ModuleDict,
    feature_mapping: Dict[str, str] = None,
    use_checkpointing: bool = False,
    checkpointing_splits: int = 2,
) -> List[torch.Tensor]:
    """
    Args:
        input_type (AttrDict): whether the model input should be RGB or BGR or LAB
        interpolate_out_feat_key_name (str): what feature dimensions should be
            used to interpolate the mask
        remove_padding_before_feat_key_name (str): name of the feature block for which
            the input should have padding removed using the interpolated mask
        feat (MultiDimensionalTensor): model input
        feature_blocks (nn.ModuleDict): ModuleDict containing feature blocks in the model
        feature_mapping (Dict[str, str]): an optional correspondence table in between
            the requested feature names and the model's.

    Returns:
        out_feats: a list with the asked output features placed in the same order as in
            `out_feat_keys`.
    """
    if feature_mapping is not None:
        interpolate_out_feat_key_name = feature_mapping[interpolate_out_feat_key_name]

    model_input = transform_model_input_data_type(feat.tensor, input_type)
    out = get_trunk_forward_outputs(
        feat=model_input,
        out_feat_keys=[interpolate_out_feat_key_name],
        feature_blocks=feature_blocks,
        use_checkpointing=use_checkpointing,
        checkpointing_splits=checkpointing_splits,
    )
    # mask is of shape N x H x W and has 1.0 value for places with padding
    # we interpolate the mask spatially to N x out.shape[-2] x out.shape[-1].
    interp_mask = F.interpolate(feat.mask[None].float(), size=out[0].shape[-2:]).to(
        torch.bool
    )[0]

    # we want to iterate over the rest of the feature blocks now
    _, max_out_feat = parse_out_keys_arg(
        [interpolate_out_feat_key_name], list(feature_blocks.keys())
    )
    for i, (feature_name, feature_block) in enumerate(feature_blocks.items()):
        # We have already done the forward till the max_out_feat.
        # we forward through rest of the blocks now.
        if i >= (max_out_feat + 1):
            if remove_padding_before_feat_key_name and (
                feature_name == remove_padding_before_feat_key_name
            ):
                # negate the mask so that the padded locations have 0.0 and the non-padded
                # locations have 1.0. This is used to extract the h, w of the original tensors.
                interp_mask = (~interp_mask).chunk(len(feat.image_sizes))
                tensors = out[0].chunk(len(feat.image_sizes))
                res = []
                for i, tensor in enumerate(tensors):
                    w = torch.sum(interp_mask[i][0], dim=0)[0]
                    h = torch.sum(interp_mask[i][0], dim=1)[0]
                    res.append(feature_block(tensor[:, :, :w, :h]))
                out[0] = torch.cat(res)
            else:
                out[0] = feature_block(out[0])
    return out


def get_trunk_forward_outputs(
    feat: torch.Tensor,
    out_feat_keys: List[str],
    feature_blocks: nn.ModuleDict,
    feature_mapping: Dict[str, str] = None,
    use_checkpointing: bool = False,
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
    # whether the feature is requested to be returned.
    for i, (feature_name, feature_block) in enumerate(feature_blocks.items()):
        # The last chunk has to be non-volatile
        if use_checkpointing and i < len(feature_blocks) - 1:
            # Un-freeze the running stats in any BN layer
            for m in filter(lambda x: isinstance(x, _bn_cls), feature_block.modules()):
                m.track_running_stats = m.training

            feat = checkpoint(feature_block, feat, use_reentrant=True)

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


def lecun_normal_init(tensor, fan_in):
    trunc_normal_(tensor, std=math.sqrt(1 / fan_in))


# Contains code from https://github.com/rwightman/pytorch-image-models
# and https://github.com/facebookresearch/deit/blob/main/models.py, modified by
# Matthew # Leavitt (ito@fb.com, matthew.l.leavitt@gmail.com) and Vedanuj
# Goswami (vedanuj@fb.com).
# trunc_normal_ and _no_grad_trunc_normal_ from:
# https://github.com/rwightman/pytorch-image-models/blob/678ba4e0a2c0b52c5e7b2ec0ba689399840282ee/timm/models/layers/weight_init.py # NOQA
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Supposedly should be available in PyTorch soon. Replace when available.
    Fills the input Tensor with values drawn
    from a truncated normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


# Contains code from https://github.com/rwightman/pytorch-image-models
# and https://github.com/facebookresearch/deit/blob/main/models.py, modified by
# Matthew # Leavitt (ito@fb.com, matthew.l.leavitt@gmail.com) and Vedanuj
# Goswami (vedanuj@fb.com).
# Standardized convolution (Conv2d with Weight Standardization), as used in
# the paper, Big Transfer (BiT): General Visual Representation Learning -
# https://arxiv.org/abs/1912.11370
class StandardizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(StandardizedConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


# drop_path and DropPath modified from
# https://github.com/facebookresearch/deit/blob/main/models.py
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path
    of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
