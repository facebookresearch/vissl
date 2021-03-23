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
from classy_vision.models.regnet import (
    ActivationType,
    BlockType,
    RegNetParams,
    ResBasicBlock,
    ResBottleneckBlock,
    ResStemCifar,
    ResStemIN,
    SimpleStemIN,
    StemType,
    VanillaBlock,
)
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP, auto_wrap_bn
from fairscale.nn.wrap import enable_wrap, wrap
from vissl.models.model_helpers import (
    Flatten,
    get_trunk_forward_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk
from vissl.utils.hydra_config import AttrDict


def fsdp_wrapper(module, **kwargs):
    """Customer wrapper that does FSDP + checkpoint at the same time."""
    # TODO (Min): enable checkpoint_wrapper
    return FSDP(module, **kwargs)


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)


def init_weights(module):
    """Helper function to init weights for a given block."""
    # Performs ResNet-style weight initialization
    n = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # Note that there is no bias due to BN
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            n += 1
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
            n += 1
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()
            n += 1
    assert n > 0, (
        "Detecting bugs where we call init_weights in the wrong place "
        "and not end up finding any modules to init"
    )


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        model_config,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: nn.Module,
        activation: nn.Module,
        bot_mul: float,
        group_width: int,
        params: "RegNetParams",
        stage_index: int = 0,
    ):
        super().__init__()
        self.stage_depth = 0

        fsdp_config = {
            "wrapper_cls": fsdp_wrapper,
        }
        fsdp_config.update(model_config.FSDP_CONFIG)
        for i in range(depth):
            # Make a block and move it to cuda since shard-as-we-build of FSDP needs
            # cuda to do dist.all_gather() call.
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                params.bn_epsilon,
                params.bn_momentum,
                activation,
                bot_mul,
                group_width,
                params.se_ratio,
            ).cuda()
            # Init weight before wrapping and sharding.
            init_weights(block)

            # Now, wrap it with fsdp+checkpoint, which will perform the sharding.
            block = auto_wrap_bn(block)
            with enable_wrap(**fsdp_config):
                block = wrap(block)

            self.stage_depth += block.depth
            self.add_module(f"block{stage_index}-{i}", block)


@register_model_trunk("regnet_fsdp")
class RegNetFSDP(FSDP):
    """
    Wrap the entire trunk since we need to load checkpoint before
    train_fsdp_task.py wrapping happens.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        module = _RegNetFSDP(model_config, model_name)
        super().__init__(module, **model_config.FSDP_CONFIG)


class _RegNetFSDP(nn.Module):
    """
    Similar to RegNet trunk, but with FSDP enabled.

    Later, FSDP + vision model is ready, this can be migrated into classy_vision.
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super().__init__()
        self.model_config = model_config

        assert model_config.INPUT_TYPE in ["rgb", "bgr"], "Input type not supported"
        trunk_config = model_config.TRUNK.TRUNK_PARAMS.REGNET

        assert "name" not in trunk_config, "Please specify the RegNet Params dictionary"

        ################################################################################

        params = RegNetParams(
            depth=trunk_config["depth"],
            w_0=trunk_config["w_0"],
            w_a=trunk_config["w_a"],
            w_m=trunk_config["w_m"],
            group_w=trunk_config["group_width"],
            stem_type=trunk_config.get("stem_type", "simple_stem_in").upper(),
            stem_width=trunk_config.get("stem_width", 32),
            block_type=trunk_config.get("block_type", "res_bottleneck_block").upper(),
            activation_type=trunk_config.get("activation_type", "relu").upper(),
            use_se=trunk_config.get("use_se", True),
            se_ratio=trunk_config.get("se_ratio", 0.25),
            bn_epsilon=trunk_config.get("bn_epsilon", 1e-05),
            bn_momentum=trunk_config.get("bn_momentum", 0.1),
        )

        # We need all workers (on all nodes) to have the same weights.
        # Unlike DDP, FSDP does not sync weights using rank 0 on start.
        # Therefore, we init stem and trunk_output below within the seed context.
        #
        # TODO (Min): we can make this seed coming from the config or env.
        stem = None
        trunk_output = None
        with set_torch_seed(0):
            # Ad hoc stem
            #
            # Important: do NOT retain modules in self.stem or self.trunk_output. It may
            # seem to be harmless, but it appears that autograd will result in computing
            # grads in different order. Different ordering can cause deterministic OOM,
            # even when the peak memory otherwise is only 24GB out of 32GB.
            #
            # When debugging this, it is not enough to just dump the total module
            # params. You need to diff the module string representations.
            activation = {
                ActivationType.RELU: nn.ReLU(False),  # params.relu_in_place
            }[params.activation_type]

            stem = {
                StemType.RES_STEM_CIFAR: ResStemCifar,
                StemType.RES_STEM_IN: ResStemIN,
                StemType.SIMPLE_STEM_IN: SimpleStemIN,
            }[params.stem_type](
                3,
                params.stem_width,
                params.bn_epsilon,
                params.bn_momentum,
                activation,
            )
            init_weights(stem)
            stem = auto_wrap_bn(stem)

            # Instantiate all the AnyNet blocks in the trunk
            block_fun = {
                BlockType.VANILLA_BLOCK: VanillaBlock,
                BlockType.RES_BASIC_BLOCK: ResBasicBlock,
                BlockType.RES_BOTTLENECK_BLOCK: ResBottleneckBlock,
            }[params.block_type]

            current_width = params.stem_width

            self.trunk_depth = 0

            blocks = []

            for i, (width_out, stride, depth, bot_mul, group_width) in enumerate(
                params.get_expanded_params()
            ):
                blocks.append(
                    (
                        f"block{i+1}",
                        AnyStage(
                            model_config,
                            current_width,
                            width_out,
                            stride,
                            depth,
                            block_fun,
                            activation,
                            bot_mul,
                            group_width,
                            params,
                            stage_index=i + 1,
                        ),
                    )
                )

                self.trunk_depth += blocks[-1][1].stage_depth

                current_width = width_out

            trunk_output = nn.Sequential(OrderedDict(blocks))

        ################################################################################

        # Now map the models to the structure we want to expose for SSL tasks
        # The upstream RegNet model is made of :
        # - `stem`
        # - n x blocks in trunk_output, named `block1, block2, ..`

        # We're only interested in the stem and successive blocks
        # everything else is not picked up on purpose
        feature_blocks: List[Tuple[str, nn.Module]] = []

        # - get the stem
        feature_blocks.append(("conv1", stem))

        # - get all the feature blocks
        for k, v in trunk_output.named_children():
            assert k.startswith("block"), f"Unexpected layer name {k}"
            block_index = len(feature_blocks) + 1
            feature_blocks.append((f"res{block_index}", v))

        # - finally, add avgpool and flatten.
        feature_blocks.append(("avgpool", nn.AdaptiveAvgPool2d((1, 1))))
        feature_blocks.append(("flatten", Flatten(1)))

        self._feature_blocks = nn.ModuleDict(feature_blocks)

    def forward(self, x, out_feat_keys: List[str] = None) -> List[torch.Tensor]:
        model_input = transform_model_input_data_type(x, self.model_config)
        return get_trunk_forward_outputs(
            feat=model_input,
            out_feat_keys=out_feat_keys,
            feature_blocks=self._feature_blocks,
            # FSDP has its own activation checkpoint method. So disabling vissl's
            # method here.
            use_checkpointing=False,
            checkpointing_splits=0,
        )
