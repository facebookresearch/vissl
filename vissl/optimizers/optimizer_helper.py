# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch.nn as nn
from fvcore.common.param_scheduler import ParamScheduler
from vissl.config import AttrDict
from vissl.optimizers.param_groups.registry import get_param_group_constructor


def get_optimizer_param_groups(
    model: nn.Module,
    model_config: AttrDict,
    optimizer_config: AttrDict,
    optimizer_schedulers: Dict[str, ParamScheduler],
):
    """
    Construct the optimizer parameter groups from the optimizer config
    provided to VISSL
    """
    pg_constructor_name = optimizer_config.param_group_constructor
    constructor = get_param_group_constructor(pg_constructor_name)
    return constructor(model, model_config, optimizer_config, optimizer_schedulers)
