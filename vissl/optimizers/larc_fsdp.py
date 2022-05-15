# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict

import torch
import torch.distributed as dist
from classy_vision.optim import register_optimizer, SGD
from vissl.utils.fsdp_utils import get_global_group


@register_optimizer("sgd_fsdp")
class SGD_FSDP(SGD):
    """
    A version of the SGD optimizer whose options (such as LARC) work with FSDP
    """

    def __init__(
        self,
        larc_config: Dict[str, Any] = None,
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_larc: bool = False,
    ):
        super().__init__(
            larc_config=larc_config,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_larc=False,
        )
        self._use_fsdp_larc = use_larc

    def prepare(self, param_groups):
        super().prepare(param_groups)
        if self._use_fsdp_larc:
            self.optimizer = LARC_FSDP(
                optimizer=self.optimizer, distributed_norm=True, **self._larc_config
            )


class LARC_FSDP:
    """
    A version of the LARC optimizer which works with FSDP:
    https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        distributed_norm:
            if True compute the norm in a distributed fashion
            if False, revert to the same computations as APEX LARC
        trust_coefficient:
            Trust coefficient for calculating the lr.
            See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC.
            If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter.
            If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(
        self,
        optimizer,
        distributed_norm: bool,
        trust_coefficient: float = 0.02,
        clip: bool = True,
        eps: float = 1e-8,
    ):
        self.optim = optimizer
        self.distributed_norm = distributed_norm
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        weight_decays = []
        with torch.no_grad():

            # Compute the parameter norms and gradient norms that are
            # required to find the adaptative_lr of each parameter group
            if self.distributed_norm:
                param_norms, grad_norms = self._compute_distributed_norms(
                    self.optim.param_groups
                )
            else:
                param_norms, grad_norms = self._compute_local_norms(
                    self.optim.param_groups
                )

            param_group_count = 0
            for group in self.optim.param_groups:

                # absorb weight decay control from optimizer
                weight_decay = group["weight_decay"] if "weight_decay" in group else 0
                weight_decays.append(weight_decay)
                group["weight_decay"] = 0

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_norm = param_norms[param_group_count]
                    grad_norm = grad_norms[param_group_count]
                    param_group_count += 1

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = (
                            self.trust_coefficient
                            * (param_norm)
                            / (grad_norm + param_norm * weight_decay + self.eps)
                        )

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr
                            # it is equal to `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group["lr"], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()

        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[i]

    @staticmethod
    def _compute_local_norms(param_groups):
        """
        Compute the parameter and gradient norms for each parameter group
        """
        param_norms = []
        grad_norms = []
        for group in param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    param_norms.append(torch.norm(p.data))
                    grad_norms.append(torch.norm(p.grad.data))
        return param_norms, grad_norms

    @staticmethod
    def _compute_distributed_norms(param_groups):
        """
        Compute the parameter and gradient norms for each parameter group

        If the parameters are sharded, this require reassembling the parameters
        of each parameter group to compute the norm.

        The algorithm below makes use of all_reduce instead of all_gather,
        profiting from the fact that the norm can be decomposed into:
        - a sum of squares locally
        - all_reduced to a global sum of squares
        - and then locally apply a square root

        Due to the non-associativity of float operations, this computation will
        lead to do different norms compared to the native implementation, but
        has a much lower impact on communication:

        ```
        def all_gather_norm(data):
            tensor_list = [torch.zeros_like(data) for _ in range(get_global_group().size())]
            dist.all_gather(tensor_list, data)
            tensor_list = torch.cat(tensor_list, dim=0)
            return torch.norm(tensor_list)

        param_norms = []
        grad_norms = []
        for group in param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    param_norms.append(all_gather_norm(p.data))
                    grad_norms.append(all_gather_norm(p.grad.data))
        return param_norms, grad_norms
        ```
        """
        param_squares = []
        grad_squares = []
        for group in param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    param_squares.append((p.data**2).sum())
                    grad_squares.append((p.grad.data**2).sum())
        all_squared = torch.stack(param_squares + grad_squares)
        dist.all_reduce(all_squared, group=get_global_group())
        all_squared = all_squared**0.5
        param_norms = all_squared[: len(param_squares)]
        grad_norms = all_squared[len(param_squares) :]
        return param_norms, grad_norms
