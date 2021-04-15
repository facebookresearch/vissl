# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP, auto_wrap_bn
from fairscale.nn.wrap import enable_wrap, wrap
from vissl.config import AttrDict
from vissl.models.model_helpers import (
    Flatten,
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk
from vissl.utils.misc import set_torch_seed


def fsdp_wrapper(module, **kwargs):
    """Customer wrapper that does FSDP + checkpoint at the same time."""
    # TODO (Min): enable checkpoint_wrapper
    return FSDP(module, **kwargs)


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


class RegnetBlocksFactory:
    """
    This is the basic RegNet construction class. It constructs the RegNet
    model stem, block creation. The RegNetParams / AnyNetParams are used
    to configure the model.
    """

    def __init__(self, seed: int):
        self.seed = seed

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

        # set stem seeds
        with set_torch_seed(self.seed):
            init_weights(stem)
            self.seed += 1
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
        ).cuda()
        with set_torch_seed(self.seed):
            init_weights(block)
            self.seed += 1
        return block


class RegnetFSDPBlocksFactory(RegnetBlocksFactory):
    """
    Simply wrap the RegnetBlocksFactory with the FSDP
    This takes care of wrapping BN properly,
    initializing the weights etc.
    """

    def __init__(self, seed: int, fsdp_config):
        super().__init__(seed)
        self.fsdp_config = fsdp_config

    def create_stem(self, params: Union[RegNetParams, AnyNetParams]):
        stem = super().create_stem(params)
        stem = auto_wrap_bn(stem, single_rank_pg=False)
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
        block = auto_wrap_bn(block, single_rank_pg=False)
        with enable_wrap(wrapper_cls=fsdp_wrapper, **self.fsdp_config):
            block = wrap(block)
        return block


class AnyStage(nn.Sequential):
    """
    AnyNet stage (sequence of blocks w/ the same output shape).
    """

    def __init__(
        self,
        factory: RegnetBlocksFactory,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        group_width: int,
        bottleneck_multiplier: float,
        params: Union[RegNetParams, AnyNetParams],
        stage_index: int = 0,
    ):
        super().__init__()
        self.stage_depth = 0
        for i in range(depth):
            block = factory.create_block(
                width_in=width_in if i == 0 else width_out,
                width_out=width_out,
                stride=stride if i == 0 else 1,
                params=params,
                group_width=group_width,
                bottleneck_multiplier=bottleneck_multiplier,
            )
            self.stage_depth += block.depth
            self.add_module(f"block{stage_index}-{i}", block)


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
    current_width = params.stem_width
    trunk_depth = 0
    blocks = []
    for i, (width_out, stride, depth, group_width, bottleneck_multiplier) in enumerate(
        params.get_expanded_params()
    ):
        blocks.append(
            (
                f"block{i + 1}",
                AnyStage(
                    factory=factory,
                    width_in=current_width,
                    width_out=width_out,
                    stride=stride,
                    depth=depth,
                    group_width=group_width,
                    bottleneck_multiplier=bottleneck_multiplier,
                    params=params,
                    stage_index=i + 1,
                ),
            )
        )
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


@register_model_trunk("regnet_2")
class RegNet3(nn.Module):
    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.model_config = model_config
        self.use_activation_checkpointing = (
            model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.activation_checkpointing_splits = (
            model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )
        self._feature_blocks, self.trunk_depth = create_regnet_feature_blocks(
            factory=RegnetBlocksFactory(seed=self.model_config._MODEL_INIT_SEED),
            model_config=model_config,
        )

    def forward(self, x, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        model_input = transform_model_input_data_type(x, self.model_config)
        return get_trunk_forward_outputs(
            feat=model_input,
            out_feat_keys=out_feat_keys,
            feature_blocks=self._feature_blocks,
            use_checkpointing=self.use_activation_checkpointing,
            checkpointing_splits=self.activation_checkpointing_splits,
        )


@register_model_trunk("regnet_fsdp_2")
class RegNetFSDP3(FSDP):
    """
    Wrap the entire trunk since we need to load checkpoint before
    train_fsdp_task.py wrapping happens.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        module = _RegNetFSDP3(model_config, model_name).cuda()
        super().__init__(module, **model_config.FSDP_CONFIG)


class _RegNetFSDP3(nn.Module):
    """
    Similar to RegNet trunk, but with FSDP enabled.

    Later, FSDP + vision model is ready, this can be migrated into classy_vision.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.model_config = model_config
        self._feature_blocks, self.trunk_depth = create_regnet_feature_blocks(
            factory=RegnetFSDPBlocksFactory(
                seed=self.model_config._MODEL_INIT_SEED,
                fsdp_config=self.model_config.FSDP_CONFIG,
            ),
            model_config=model_config,
        )

    def forward(self, x, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        model_input = transform_model_input_data_type(x, self.model_config)
        return get_trunk_forward_outputs(
            feat=model_input,
            out_feat_keys=out_feat_keys,
            feature_blocks=self._feature_blocks,
            # FSDP has its own activation checkpoint method: disable vissl's method here.
            use_checkpointing=False,
            checkpointing_splits=0,
        )
