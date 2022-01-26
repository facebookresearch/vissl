# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom version of RegNet for FSDP.

Eventually this would go upstream to classy but since we are still
developing on both vissl and fairscale sides, it is much easier to
eep a custom version. For large scale training, FSDP acts like a
DDP replacement. However, due to implementation details, special
cares (like dealing with BN) need to be taken in the model. Therefore,
we keep this version here. We aim to ensure the model convergence and
accuracy is minimally impacted to the extend allowed by FSDP
and target training speed considerations.
"""

import math
from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from classy_vision.generic.util import get_torch_version
from classy_vision.models.anynet import (
    ActivationType,
    AnyNetParams,
    BlockType,
    ResBasicBlock,
    ResBottleneckBlock,
    ResBottleneckLinearBlock,
    ResStemCifar,
    ResStemIN,
    SimpleStemIN,
    StemType,
    VanillaBlock,
)
from classy_vision.models.regnet import RegNetParams
from fairscale.nn import checkpoint_wrapper
from vissl.config import AttrDict
from vissl.data.collators.collator_helper import MultiDimensionalTensor
from vissl.models.model_helpers import (
    Flatten,
    get_trunk_forward_outputs,
    get_tunk_forward_interpolated_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk
from vissl.utils.fsdp_utils import auto_wrap_big_layers, fsdp_auto_wrap_bn, fsdp_wrapper
from vissl.utils.misc import set_torch_seed


def init_weights(module):
    """
    Helper function to init weights for a given block.
    Inspired by classy_vision.models.regnet.RegNet.init_weights
    """
    num_initialized = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # Note that there is no bias due to BN
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            num_initialized += 1
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
            num_initialized += 1
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()
            num_initialized += 1
    assert num_initialized > 0, (
        "Detecting bugs where we call init_weights in the wrong place "
        "and not end up finding any modules to init"
    )


class AnyStage(nn.Sequential):
    """
    AnyNet stage (sequence of blocks w/ the same output shape).
    """

    def __init__(self):
        super().__init__()
        self.stage_depth = 0


class RegnetBlocksFactory:
    """
    This is the basic RegNet construction class. It constructs the RegNet
    model stem, block creation. The RegNetParams / AnyNetParams are used
    to configure the model.
    """

    def create_stem(self, params: Union[RegNetParams, AnyNetParams]):
        # get the activation
        silu = None if get_torch_version() < [1, 7] else nn.SiLU()
        activation = {
            ActivationType.RELU: nn.ReLU(params.relu_in_place),
            ActivationType.SILU: silu,
        }[params.activation]

        # create stem
        stem = {
            StemType.RES_STEM_CIFAR: ResStemCifar,
            StemType.RES_STEM_IN: ResStemIN,
            StemType.SIMPLE_STEM_IN: SimpleStemIN,
        }[params.stem_type](
            3, params.stem_width, params.bn_epsilon, params.bn_momentum, activation
        )
        init_weights(stem)
        return stem

    def create_block(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        params: Union[RegNetParams, AnyNetParams],
        bottleneck_multiplier: float,
        group_width: int = 1,
    ):
        # get the block constructor function to use
        block_constructor = {
            BlockType.VANILLA_BLOCK: VanillaBlock,
            BlockType.RES_BASIC_BLOCK: ResBasicBlock,
            BlockType.RES_BOTTLENECK_BLOCK: ResBottleneckBlock,
            BlockType.RES_BOTTLENECK_LINEAR_BLOCK: ResBottleneckLinearBlock,
        }[params.block_type]

        # get the activation module
        silu = None if get_torch_version() < [1, 7] else nn.SiLU()
        activation = {
            ActivationType.RELU: nn.ReLU(params.relu_in_place),
            ActivationType.SILU: silu,
        }[params.activation]

        block = block_constructor(
            width_in,
            width_out,
            stride,
            params.bn_epsilon,
            params.bn_momentum,
            activation,
            group_width,
            bottleneck_multiplier,
            params.se_ratio,
        )
        init_weights(block)
        return block

    def create_any_stage(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        group_width: int,
        bottleneck_multiplier: float,
        params: Union[RegNetParams, AnyNetParams],
        group_delimiters: List[int],
        group_checkpoint: List[bool],
        stage_index: int = 0,
    ) -> AnyStage:
        assert len(group_delimiters) == len(group_checkpoint)

        any_stage = AnyStage()
        prev_depth = 0
        for group_index, next_depth in enumerate(group_delimiters):
            block_group = nn.Sequential()
            for i in range(prev_depth, next_depth):
                block = self.create_block(
                    width_in=width_in if i == 0 else width_out,
                    width_out=width_out,
                    stride=stride if i == 0 else 1,
                    params=params,
                    group_width=group_width,
                    bottleneck_multiplier=bottleneck_multiplier,
                )
                any_stage.stage_depth += block.depth
                block_group.add_module(f"block{stage_index}-{i}", block)
            prev_depth = next_depth
            if group_checkpoint[group_index]:
                block_group = checkpoint_wrapper(block_group)
            any_stage.add_module(f"block{stage_index}-part{group_index}", block_group)
        return any_stage


class RegnetFSDPBlocksFactory(RegnetBlocksFactory):
    """
    Simply wrap the RegnetBlocksFactory with the FSDP
    This takes care of wrapping BN properly,
    initializing the weights etc.
    """

    def __init__(self, fsdp_config: AttrDict):
        super().__init__()
        self.fsdp_config = fsdp_config

    def create_stem(self, params: Union[RegNetParams, AnyNetParams]):
        stem = super().create_stem(params)
        stem = fsdp_auto_wrap_bn(stem)
        return stem

    def create_block(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        params: Union[RegNetParams, AnyNetParams],
        bottleneck_multiplier: float,
        group_width: int = 1,
    ):
        block = super().create_block(
            width_in, width_out, stride, params, bottleneck_multiplier, group_width
        )
        block = fsdp_auto_wrap_bn(block)
        if self.fsdp_config.AUTO_WRAP_THRESHOLD > 0:
            block = auto_wrap_big_layers(block, self.fsdp_config)
        block = fsdp_wrapper(module=block, **self.fsdp_config)
        return block

    def create_any_stage(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        group_width: int,
        bottleneck_multiplier: float,
        params: Union[RegNetParams, AnyNetParams],
        group_delimiters: List[int],
        group_checkpoint: List[bool],
        stage_index: int = 0,
    ):
        assert len(group_delimiters) == len(group_checkpoint)

        any_stage = AnyStage()
        prev_depth = 0
        for group_index, next_depth in enumerate(group_delimiters):
            block_group = nn.Sequential()
            for i in range(prev_depth, next_depth):
                block = self.create_block(
                    width_in=width_in if i == 0 else width_out,
                    width_out=width_out,
                    stride=stride if i == 0 else 1,
                    params=params,
                    group_width=group_width,
                    bottleneck_multiplier=bottleneck_multiplier,
                )
                any_stage.stage_depth += block.depth
                block_group.add_module(f"block{stage_index}-{i}", block)
            prev_depth = next_depth
            if group_checkpoint[group_index]:
                block_group = checkpoint_wrapper(block_group)
            block_group = fsdp_wrapper(block_group, **self.fsdp_config)
            any_stage.add_module(f"block{stage_index}-part{group_index}", block_group)
        return any_stage


def create_regnet_feature_blocks(factory: RegnetBlocksFactory, model_config):
    assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
    trunk_config = model_config.TRUNK.REGNET
    if "name" in trunk_config:
        assert (
            trunk_config["name"] == "anynet"
        ), "Please use AnyNetParams or specify RegNetParams dictionary"

    if "name" in trunk_config and trunk_config["name"] == "anynet":
        params = AnyNetParams(
            depths=trunk_config["depths"],
            widths=trunk_config["widths"],
            group_widths=trunk_config["group_widths"],
            bottleneck_multipliers=trunk_config["bottleneck_multipliers"],
            strides=trunk_config["strides"],
            stem_type=StemType[trunk_config.get("stem_type", "simple_stem_in").upper()],
            stem_width=trunk_config.get("stem_width", 32),
            block_type=BlockType[
                trunk_config.get("block_type", "res_bottleneck_block").upper()
            ],
            activation=ActivationType[trunk_config.get("activation", "relu").upper()],
            use_se=trunk_config.get("use_se", True),
            se_ratio=trunk_config.get("se_ratio", 0.25),
            bn_epsilon=trunk_config.get("bn_epsilon", 1e-05),
            bn_momentum=trunk_config.get("bn_momentum", 0.1),
        )
    else:
        params = RegNetParams(
            depth=trunk_config["depth"],
            w_0=trunk_config["w_0"],
            w_a=trunk_config["w_a"],
            w_m=trunk_config["w_m"],
            group_width=trunk_config["group_width"],
            bottleneck_multiplier=trunk_config.get("bottleneck_multiplier", 1.0),
            stem_type=StemType[trunk_config.get("stem_type", "simple_stem_in").upper()],
            stem_width=trunk_config.get("stem_width", 32),
            block_type=BlockType[
                trunk_config.get("block_type", "res_bottleneck_block").upper()
            ],
            activation=ActivationType[trunk_config.get("activation", "relu").upper()],
            use_se=trunk_config.get("use_se", True),
            se_ratio=trunk_config.get("se_ratio", 0.25),
            bn_epsilon=trunk_config.get("bn_epsilon", 1e-05),
            bn_momentum=trunk_config.get("bn_momentum", 0.1),
        )

    # Ad hoc stem
    #
    # Important: do NOT retain modules in self.stem or self.trunk_output. It may
    # seem to be harmless, but it appears that autograd will result in computing
    # grads in different order. Different ordering can cause deterministic OOM,
    # even when the peak memory otherwise is only 24GB out of 32GB.
    #
    # When debugging this, it is not enough to just dump the total module
    # params. You need to diff the module string representations.
    stem = factory.create_stem(params)

    # Instantiate all the AnyNet blocks in the trunk
    current_width, trunk_depth, blocks = params.stem_width, 0, []
    for i, (width_out, stride, depth, group_width, bottleneck_multiplier) in enumerate(
        params.get_expanded_params()
    ):
        # Starting from 1
        stage_index = i + 1

        # Identify where the block groups start and end, and whether they should
        # be surrounded by activation checkpoints
        # A block group is a group of block that is surrounded by a FSDP wrapper
        # and optionally an activation checkpoint wrapper
        with_checkpointing = (
            model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        all_group_delimiters = trunk_config.get("stage_checkpoints", [])
        all_group_checkpoint = trunk_config.get("stage_checkpointing", [])
        group_delimiters = (
            all_group_delimiters[i] if len(all_group_delimiters) > i else []
        )
        group_checkpoint = (
            all_group_checkpoint[i] if len(all_group_checkpoint) > i else []
        )
        if not group_checkpoint:
            group_checkpoint = [with_checkpointing] * len(group_delimiters)
        assert len(group_delimiters) == len(group_checkpoint)

        assert (
            sorted(group_delimiters) == group_delimiters
        ), "Checkpoint boundaries should be sorted"
        if not group_delimiters:
            # No delimiters means one group but no activation checkpointing
            # for this group (even if USE_ACTIVATION_CHECKPOINTING is set)
            group_delimiters.append(depth)
            group_checkpoint.append(False)
        elif group_delimiters[-1] != depth:
            # Complete missing checkpoints at the end (user can give only
            # the intermediate checkpoints to avoid repetitions)
            group_delimiters.append(depth)
            group_checkpoint.append(with_checkpointing)

        # Create the stage from the description of the block and the size of
        # the block groups that compose this stage, then add it to the trunk
        new_stage = factory.create_any_stage(
            width_in=current_width,
            width_out=width_out,
            stride=stride,
            depth=depth,
            group_width=group_width,
            bottleneck_multiplier=bottleneck_multiplier,
            params=params,
            stage_index=stage_index,
            group_delimiters=group_delimiters,
            group_checkpoint=group_checkpoint,
        )
        blocks.append((f"block{stage_index}", new_stage))
        trunk_depth += blocks[-1][1].stage_depth
        current_width = width_out

    trunk_output = nn.Sequential(OrderedDict(blocks))

    ################################################################################

    # Now map the models to the structure we want to expose for SSL tasks
    # The upstream RegNet model is made of :
    # - `stem`
    # - n x blocks in trunk_output, named `block1, block2, ..`
    # We're only interested in the stem and successive blocks
    # everything else is not picked up on purpose
    feature_blocks: List[Tuple[str, nn.Module]] = [("conv1", stem)]
    for k, v in trunk_output.named_children():
        assert k.startswith("block"), f"Unexpected layer name {k}"
        block_index = len(feature_blocks) + 1
        feature_blocks.append((f"res{block_index}", v))
    feature_blocks.append(("avgpool", nn.AdaptiveAvgPool2d((1, 1))))
    feature_blocks.append(("flatten", Flatten(1)))
    return nn.ModuleDict(feature_blocks), trunk_depth


@register_model_trunk("regnet_v2")
class RegNetV2(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.model_config = model_config
        self.seed = self.model_config._MODEL_INIT_SEED
        self.use_activation_checkpointing = (
            model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.activation_checkpointing_splits = (
            model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )
        with set_torch_seed(self.seed):
            self._feature_blocks, self.trunk_depth = create_regnet_feature_blocks(
                factory=RegnetBlocksFactory(), model_config=model_config
            )

    def forward(self, x, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        if isinstance(x, MultiDimensionalTensor):
            out = get_tunk_forward_interpolated_outputs(
                input_type=self.model_config.INPUT_TYPE,
                interpolate_out_feat_key_name="res5",
                remove_padding_before_feat_key_name="avgpool",
                feat=x,
                feature_blocks=self._feature_blocks,
                use_checkpointing=self.use_activation_checkpointing,
                checkpointing_splits=self.activation_checkpointing_splits,
            )
        else:
            model_input = transform_model_input_data_type(
                x, self.model_config.INPUT_TYPE
            )
            out = get_trunk_forward_outputs(
                feat=model_input,
                out_feat_keys=out_feat_keys,
                feature_blocks=self._feature_blocks,
                use_checkpointing=self.use_activation_checkpointing,
                checkpointing_splits=self.activation_checkpointing_splits,
            )
        return out


@register_model_trunk("regnet_fsdp")
def RegNetFSDP(model_config: AttrDict, model_name: str):
    """
    Wrap the entire trunk since we need to load checkpoint before
    train_fsdp_task.py wrapping happens.
    """
    module = _RegNetFSDP(model_config, model_name)
    return fsdp_wrapper(module, **model_config.FSDP_CONFIG)


class _RegNetFSDP(nn.Module):
    """
    Similar to RegNet trunk, but with FSDP enabled.

    Later, FSDP + vision model is ready, this can be migrated into classy_vision.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.model_config = model_config
        self.seed = self.model_config._MODEL_INIT_SEED
        self.use_activation_checkpointing = (
            model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        with set_torch_seed(self.seed):
            self._feature_blocks, self.trunk_depth = create_regnet_feature_blocks(
                factory=RegnetFSDPBlocksFactory(
                    fsdp_config=self.model_config.FSDP_CONFIG
                ),
                model_config=self.model_config,
            )

    def forward(self, x, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        if isinstance(x, MultiDimensionalTensor):
            out = get_tunk_forward_interpolated_outputs(
                input_type=self.model_config.INPUT_TYPE,
                interpolate_out_feat_key_name="res5",
                remove_padding_before_feat_key_name="avgpool",
                feat=x,
                feature_blocks=self._feature_blocks,
                # FSDP has its own activation checkpoint method: disable vissl's method here.
                use_checkpointing=False,
                checkpointing_splits=0,
            )
        else:
            model_input = transform_model_input_data_type(
                x, self.model_config.INPUT_TYPE
            )
            out = get_trunk_forward_outputs(
                feat=model_input,
                out_feat_keys=out_feat_keys,
                feature_blocks=self._feature_blocks,
                # FSDP has its own activation checkpoint method: disable vissl's method here.
                use_checkpointing=False,
                checkpointing_splits=0,
            )
        return out
