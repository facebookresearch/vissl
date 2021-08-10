# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch
from classy_vision.optim import ClassyOptimizer, register_optimizer
from torch import optim


@register_optimizer("lars")
class LARS(ClassyOptimizer):
    def __init__(
        self,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eta: float = 0.001,
        exclude_bias_and_norm: bool = False,
    ):
        super(LARS, self).__init__()

        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._eta = eta
        self._exclude_bias_and_norm = exclude_bias_and_norm

        self.optimizer = None

    def prepare(self, param_groups):
        self.optimizer = _LARS(
            param_groups,
            lr=self._lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
            eta=self._eta,
            exclude_bias_and_norm=self._exclude_bias_and_norm,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LARS":
        # Default params
        config.setdefault("lr", 0.1)
        config.setdefault("momentum", 0.9)
        config.setdefault("weight_decay", 0.0)
        config.setdefault("eta", 0.001)
        config.setdefault("exclude_bias_and_norm", False)

        assert (
            config["momentum"] >= 0.0
            and config["momentum"] < 1.0
            and type(config["momentum"]) == float
        ), "Config must contain a 'momentum' in [0, 1) for SGD optimizer"

        return cls(
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            eta=config["eta"],
            exclude_bias_and_norm=config["exclude_bias_and_norm"],
        )


class _LARS(optim.Optimizer):
    """
    From: https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L218

    Main differences:
        - we add a `exclude_bias_and_norm` parameter to filter out biases from the LARS adaptive LR
        - we remove the filter on weight decay as it is not needed in VISSL. See `get_optimizer_param_groups`
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eta: float = 0.001,
        exclude_bias_and_norm: bool = False,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eta": eta,
            "exclude": exclude_bias_and_norm,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _exclude_bias_and_norm(p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                dp = dp.add(p, alpha=g["weight_decay"])

                if not g["exclude"] or not self._exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)

                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])
