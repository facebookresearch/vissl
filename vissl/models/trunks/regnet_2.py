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
from typing import List, Tuple

import torch
import torch.nn as nn
from classy_vision.models.anynet import (
    ResBasicBlock,
    ResBottleneckBlock,
    ResStemCifar,
    ResStemIN,
    SimpleStemIN,
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
    STEM_TYPES = {
        "RES_STEM_CIFAR": ResStemCifar,
        "RES_STEM_IN": ResStemIN,
        "SIMPLE_STEM_IN": SimpleStemIN,
    }

    BLOCK_TYPES = {
        "VANILLA_BLOCK": VanillaBlock,
        "RES_BASIC_BLOCK": ResBasicBlock,
        "RES_BOTTLENECK_BLOCK": ResBottleneckBlock,
    }

    ACTIVATION_TYPES = {"RELU": lambda: nn.ReLU(inplace=True)}

    def __init__(self, seed: int):
        self.seed = seed

    def create_stem(self, params: RegNetParams):
        activation = self.ACTIVATION_TYPES[params.activation]()
        stem = self.STEM_TYPES[params.stem_type](
            3, params.stem_width, params.bn_epsilon, params.bn_momentum, activation
        )
        with set_torch_seed(self.seed):
            init_weights(stem)
            self.seed += 1
        return stem

    def create_block(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        params: RegNetParams,
        bot_mul: float,
        group_width: int = 1,
    ):
        block_constructor = self.BLOCK_TYPES[params.block_type.upper()]
        activation = self.ACTIVATION_TYPES[params.activation]()
        block = block_constructor(
            width_in,
            width_out,
            stride,
            params.bn_epsilon,
            params.bn_momentum,
            activation,
            bot_mul,
            group_width,
            params.se_ratio,
        ).cuda()
        with set_torch_seed(self.seed):
            init_weights(block)
            self.seed += 1
        return block


class RegnetFSDPBlocksFactory(RegnetBlocksFactory):
    def __init__(self, seed: int, fsdp_config):
        super().__init__(seed)
        self.fsdp_config = fsdp_config

    def create_stem(self, params: RegNetParams):
        stem = super().create_stem(params)
        stem = auto_wrap_bn(stem, single_rank_pg=False)
        return stem

    def create_block(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        params: "RegNetParams",
        bot_mul: float,
        group_width: int = 1,
    ):
        block = super().create_block(
            width_in, width_out, stride, params, bot_mul, group_width
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
        bot_mul: float,
        group_width: int,
        params: "RegNetParams",
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
                bot_mul=bot_mul,
                group_width=group_width,
            )
            self.stage_depth += block.depth
            self.add_module(f"block{stage_index}-{i}", block)


def create_regnet_feature_blocks(factory: RegnetBlocksFactory, model_config):
    assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
    trunk_config = model_config.TRUNK.TRUNK_PARAMS.REGNET
    assert "name" not in trunk_config, "Please specify the RegNet Params dictionary"

    params = RegNetParams(
        depth=trunk_config["depth"],
        w_0=trunk_config["w_0"],
        w_a=trunk_config["w_a"],
        w_m=trunk_config["w_m"],
        group_width=trunk_config["group_width"],
        stem_type=trunk_config.get("stem_type", "simple_stem_in").upper(),
        stem_width=trunk_config.get("stem_width", 32),
        block_type=trunk_config.get("block_type", "res_bottleneck_block").upper(),
        activation=trunk_config.get("activation_type", "relu").upper(),
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
    for i, (width_out, stride, depth, bot_mul, group_width) in enumerate(
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
                    bot_mul=bot_mul,
                    group_width=group_width,
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
            use_checkpointing=self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING,
            checkpointing_splits=self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS,
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
