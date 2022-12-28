# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from vissl.config import AttrDict
from vissl.models.heads import register_model_head
from vissl.models.model_helpers import trunc_normal_


@register_model_head("msn_head")
class MSNHead(nn.Module):
    """
    Specific head for training MSN (https://arxiv.org/pdf/2204.07141.pdf).
    """

    def __init__(
        self,
        model_config: AttrDict,
        in_dim: int,
        num_prototypes: int,
        use_bn: bool = False,
        hidden_dim: int = 2048,
        output_dim: int = 256,
        freeze_prototypes: bool = False,
        label_smoothing: float = 0.0,
        temperature: float = 0.1,
        return_query: bool = False,  # Used for evaluation of representations
    ):
        super().__init__()
        self.temperature = temperature
        self.return_query = return_query

        # Create the projection head
        fc = OrderedDict([])
        fc["fc1"] = torch.nn.Linear(in_dim, hidden_dim)
        if use_bn:
            fc["bn1"] = torch.nn.BatchNorm1d(hidden_dim)
        fc["gelu1"] = torch.nn.GELU()
        fc["fc2"] = torch.nn.Linear(hidden_dim, hidden_dim)
        if use_bn:
            fc["bn2"] = torch.nn.BatchNorm1d(hidden_dim)
        fc["gelu2"] = torch.nn.GELU()
        fc["fc3"] = torch.nn.Linear(hidden_dim, output_dim)
        self.fc = torch.nn.Sequential(fc)
        self.apply(self._init_weights)

        # Instantiate the prototypes
        with torch.no_grad():
            prototypes = torch.empty(num_prototypes, output_dim)
            _sqrt_k = (1.0 / output_dim) ** 0.5
            torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)
            self.prototypes = torch.nn.parameter.Parameter(prototypes)

            # Init prototype labels
            self.register_buffer(
                "proto_labels",
                self.one_hot(
                    torch.tensor(list(range(num_prototypes))),
                    num_prototypes,
                    label_smoothing,
                ),
                persistent=False,
            )
        if not freeze_prototypes:
            prototypes.requires_grad = True

        # Softmax for the allocation to prototypes
        self.softmax = torch.nn.Softmax(dim=1)

    @staticmethod
    def one_hot(targets, num_classes, smoothing):
        off_value = smoothing / num_classes
        on_value = 1.0 - smoothing + off_value
        targets = targets.long().view(-1, 1)
        return torch.full((len(targets), num_classes), off_value).scatter_(
            1, targets, on_value
        )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        query = F.normalize(x, dim=-1, p=2)
        if self.return_query:
            return query

        supports = F.normalize(self.prototypes)
        out = self.softmax(query @ supports.T / self.temperature) @ self.proto_labels
        return out
