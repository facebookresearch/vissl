# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from vissl.config import AttrDict
from vissl.models.heads import register_model_head


@register_model_head("identity")
def IdentityHead(_model_config: AttrDict):
    return nn.Identity()
