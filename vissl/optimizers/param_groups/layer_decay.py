# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Dict

import torch.nn as nn
from fvcore.common.param_scheduler import ParamScheduler
from vissl.config import AttrDict
from vissl.optimizers.param_groups.registry import register_param_group_constructor
from vissl.optimizers.param_scheduler.layer_decay_scheduler import LayerDecayScheduler
from vissl.utils.misc import is_apex_available


# TODO -  factorize with the other list
_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,  # pytorch SyncBN
    nn.LayerNorm,
)

if is_apex_available():
    import apex

    _NORM_TYPES += (apex.parallel.SyncBatchNorm,)


class LayerDecayGroupConstructor:
    def __init__(
        self,
        layer_decay: float,
        num_layers: int,
        optimizer_schedulers,
        optimizer_config,
    ):
        super().__init__()
        self.layer_decay = layer_decay
        self.num_layers = num_layers
        if "weight_decay" in optimizer_schedulers:
            self.weight_decay_main_config = optimizer_schedulers["weight_decay"]
        else:
            self.weight_decay_main_config = optimizer_config.weight_decay
        self.all_groups = {}

    def append(
        self, layer_depth: int, with_regularisation: bool, optimizer_schedulers, param
    ):
        key = (with_regularisation, layer_depth)
        if key not in self.all_groups:
            update_interval = optimizer_schedulers["lr"].update_interval
            lr_scale = self.layer_decay ** (self.num_layers + 1 - layer_depth)
            self.all_groups[key] = {
                "params": [],
                "lr": LayerDecayScheduler(
                    optimizer_schedulers["lr"], lr_scale, update_interval
                ),
                "weight_decay": (
                    self.weight_decay_main_config if with_regularisation else 0.0
                ),
            }
        self.all_groups[key]["params"].append(param)

    def get(self):
        return list(self.all_groups.values())


def get_vit_param_depth(param_name: str, num_layers: int) -> int:
    layer_0 = {"patch_embed", "pos_embed", "cls_token"}
    if any(l in param_name for l in layer_0):
        return 0

    block_regex = re.compile(r"blocks\.([0-9]+)\.")
    match = block_regex.search(param_name)
    if match is not None:
        return int(match.group(1)) + 1
    else:
        return num_layers


def get_vit_lr_decay_param_groups(
    model: nn.Module,
    model_config: AttrDict,
    optimizer_config: AttrDict,
    optimizer_schedulers: Dict[str, ParamScheduler],
):
    layer_decay = optimizer_config.layer_wise_lr_decay.decay
    num_layers = model_config.TRUNK.VISION_TRANSFORMERS.NUM_LAYERS

    all_groups = LayerDecayGroupConstructor(
        layer_decay, num_layers, optimizer_schedulers, optimizer_config
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # we do not regularize biases nor Norm parameters
        with_regularisation = True
        if name.endswith(".bias") or "norm" in name:
            with_regularisation = False

        # Get the target layer
        if "heads" in name:
            layer_depth = num_layers + 1
        else:
            layer_depth = get_vit_param_depth(name, num_layers)

        # Add the parameter group
        all_groups.append(
            layer_depth=layer_depth,
            with_regularisation=with_regularisation,
            optimizer_schedulers=optimizer_schedulers,
            param=param,
        )

    return all_groups.get()


def get_resnet_param_depth(param_name: str, num_layers: int) -> int:
    layer_0 = {
        "_feature_blocks.conv1",
        "_feature_blocks.bn1",
    }
    if any(l in param_name for l in layer_0):
        return 0

    stage_regex = re.compile(r"\.layer([0-9]*)\.")
    match = stage_regex.search(param_name)
    if match is not None:
        return int(match.group(1))
    else:
        return num_layers + 1


def get_resnet_lr_decay_param_groups(
    model: nn.Module,
    model_config: AttrDict,
    optimizer_config: AttrDict,
    optimizer_schedulers: Dict[str, ParamScheduler],
):
    layer_decay = optimizer_config.layer_wise_lr_decay.decay
    num_layers = 4
    bn_regex = re.compile(r"\.bn[0-9]*\.")

    all_groups = LayerDecayGroupConstructor(
        layer_decay, num_layers, optimizer_schedulers, optimizer_config
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Regularize of biases or norm parameters
        with_regularisation = True
        if name.endswith(".bias"):
            with_regularisation = optimizer_config["regularize_bias"]
        elif bn_regex.search(name) is not None:
            with_regularisation = optimizer_config["regularize_bn"]

        # Get the target layer
        layer_depth = get_resnet_param_depth(name, num_layers)

        # Add the parameter group
        all_groups.append(
            layer_depth=layer_depth,
            with_regularisation=with_regularisation,
            optimizer_schedulers=optimizer_schedulers,
            param=param,
        )

    return all_groups.get()


@register_param_group_constructor("lr_decay")
def get_lr_decay_param_groups(
    model: nn.Module,
    model_config: AttrDict,
    optimizer_config: AttrDict,
    optimizer_schedulers: Dict[str, ParamScheduler],
):
    trunk_name = model_config.TRUNK.NAME
    if trunk_name == "vision_transformer":
        return get_vit_lr_decay_param_groups(
            model, model_config, optimizer_config, optimizer_schedulers
        )
    elif trunk_name == "resnet":
        return get_resnet_lr_decay_param_groups(
            model, model_config, optimizer_config, optimizer_schedulers
        )
    else:
        raise ValueError(f"Undefined LR Decay scheme for {trunk_name}")


@register_param_group_constructor("linear_eval_heads")
def get_lr_heads_param_groups(
    model: nn.Module,
    model_config: AttrDict,
    optimizer_config: AttrDict,
    optimizer_schedulers: Dict[str, ParamScheduler],
):
    """
    Parameter group that can be used in conjunction with linear evaluation
    to have different heads used different learning rates
    """

    regex_head_index = re.compile(r"heads\.([0-9]*)")

    parameter_groups = []
    lr_heads = optimizer_config.linear_eval_heads
    default_weight_decay = optimizer_config.weight_decay

    for name, module in model.named_modules():
        if "heads" not in name:
            continue

        match = regex_head_index.match(name)
        if match is not None:
            head_index = int(match.group(1))
            lr_decay = lr_heads[head_index]["lr"]
            weight_decay = lr_heads[head_index].get(
                "weight_decay", default_weight_decay
            )
            regul_bn = lr_heads[head_index].get(
                "regularize_bn", optimizer_config["regularize_bn"]
            )
        else:
            lr_decay = 1.0
            weight_decay = default_weight_decay
            regul_bn = optimizer_config["regularize_bn"]

        unregularized_params = []
        regularized_params = []

        if isinstance(module, nn.Linear) or isinstance(module, _CONV_TYPES):
            regularized_params.append(module.weight)
            if module.bias is not None:
                if optimizer_config["regularize_bias"]:
                    regularized_params.append(module.bias)
                else:
                    unregularized_params.append(module.bias)
        elif isinstance(module, _NORM_TYPES):
            if module.weight is not None:
                if regul_bn:
                    regularized_params.append(module.weight)
                else:
                    unregularized_params.append(module.weight)
            if module.bias is not None:
                if regul_bn and optimizer_config["regularize_bias"]:
                    regularized_params.append(module.bias)
                else:
                    unregularized_params.append(module.bias)

        update_interval = optimizer_schedulers["lr"].update_interval
        lr = LayerDecayScheduler(optimizer_schedulers["lr"], lr_decay, update_interval)

        if unregularized_params:
            parameter_groups.append(
                {
                    "params": unregularized_params,
                    "lr": lr,
                    "weight_decay": 0.0,
                }
            )
        if regularized_params:
            parameter_groups.append(
                {
                    "params": regularized_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )

    return parameter_groups
