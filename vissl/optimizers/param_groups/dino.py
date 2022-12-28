# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict

import torch.nn as nn
from fvcore.common.param_scheduler import ParamScheduler
from vissl.config import AttrDict
from vissl.optimizers.param_groups.registry import register_param_group_constructor


@register_param_group_constructor("dino")
def get_dino_optimizer_param_groups(
    model: nn.Module,
    model_config: AttrDict,
    optimizer_config: AttrDict,
    optimizer_schedulers: Dict[str, ParamScheduler],
):
    """
    DINO specific way of creation parameter groups
    Adapted from: https://github.com/facebookresearch/dino/blob/main/utils.py
    """
    if "weight_decay" in optimizer_schedulers:
        weight_decay_main_config = optimizer_schedulers["weight_decay"]
    else:
        weight_decay_main_config = optimizer_config.weight_decay

    regularized_names = []
    not_regularized_names = []
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or "norm" in name:
            not_regularized_names.append(name)
            not_regularized.append(param)
        else:
            regularized_names.append(name)
            regularized.append(param)
    logging.info(f"Regularized parameters: {len(regularized_names)}")
    logging.info(f"Unregularized parameters: {len(not_regularized_names)}")

    param_groups = [
        {
            "params": regularized,
            "lr": optimizer_schedulers["lr"],
            "weight_decay": weight_decay_main_config,
        },
        {
            "params": not_regularized,
            "lr": optimizer_schedulers["lr"],
            "weight_decay": 0.0,
        },
    ]
    param_groups = [pg for pg in param_groups if len(pg["params"])]
    return param_groups
