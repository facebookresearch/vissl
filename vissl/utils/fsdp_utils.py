# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist
import torch.nn as nn
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap
from fairscale.nn.data_parallel import (
    auto_wrap_bn as fairscale_auto_wrap_bn,
    FullyShardedDataParallel as FSDP,
)
from vissl.config.attr_dict import AttrDict


def fsdp_recursive_reset_lazy_init(fsdp_module: FSDP):
    """
    Before the first forward pass, an FSDP module might have been initialized
    for instance by calling load_state_dict or load_local_state_dict to
    reload a previous training checkpoint.

    This function will recursively walk though the sub-FSDP modules and
    call _reset_lazy_init on each of them.
    """
    for module in fsdp_module.modules():
        if isinstance(module, FSDP) and module._is_root is not None:
            module._reset_lazy_init()


def get_global_group():
    """
    Singleton pytorch distributed group
    Inspired by https://github.com/pytorch/fairseq
    """
    if dist.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            # ideally we could use torch.distributed.group.WORLD, but it seems
            # to cause random NCCL hangs in some cases
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None


def fsdp_auto_wrap_bn(module):
    """
    Custom Batch Normalisation FSDP auto wrapper which makes
    sure to use the global group used for all other FSDP wraps
    """
    return fairscale_auto_wrap_bn(
        module, single_rank_pg=False, process_group=get_global_group()
    )


def is_fsdp_model(model):
    if isinstance(model, FSDP):
        return True
    if hasattr(model, "trunk") and isinstance(model.trunk, FSDP):
        return True
    return False


def is_valid_fsdp_model(model: FSDP) -> bool:
    """
    Checks if a FSDP model is valid by looking at the sub-FSDP modules
    and ensuring that they do not think they are the root FSDP model
    """
    for n, m in model.named_modules():
        if isinstance(m, FSDP):
            if n != "" and m._is_root is not None:
                return False
    return True


def fsdp_wrapper(module, **kwargs):
    """
    Customer FSDP wrapper, adding the missing options
    """
    from vissl.utils.layer_memory_tracking import ProcessGroupTracker

    # Add global process group to the list of keys
    fsdp_config = dict(**kwargs)
    if "process_group" not in fsdp_config:
        fsdp_config["process_group"] = get_global_group()
    if fsdp_config.get("_TRACK_COMMUNICATIONS", False):
        fsdp_config["process_group"] = ProcessGroupTracker(fsdp_config["process_group"])

    # Remove keys that are not supported in FSDP
    for key in {"_TRACK_COMMUNICATIONS", "AUTO_WRAP_THRESHOLD", "FORCE_SYNC_CUDA"}:
        fsdp_config.pop(key, None)

    return FSDP(module, **fsdp_config)


class _FSDP_WRAPPER:
    # TODO (Quentin) - remove this hack after issue is solved:
    #  https://github.com/facebookresearch/fairscale/issues/649
    def __new__(cls, module, **kwargs):
        return fsdp_wrapper(module, **kwargs)


class _BigConvAutoWrapPolicy:
    """
    Wrap convolution layers bigger than the provided number of parameters
    """

    def __init__(self, threshold: int):
        self.threshold = threshold

    def __call__(
        self, module: nn.Module, recurse: bool, unwrapped_params: int, **kwargs
    ):
        is_large = unwrapped_params >= self.threshold
        force_leaf_modules = default_auto_wrap_policy.FORCE_LEAF_MODULES
        if recurse:
            # We should recurse if the module is big enough but not in force_leaf_modules.
            return is_large and not isinstance(module, tuple(force_leaf_modules))
        else:
            # If we are not recursing, we should wrap but not the exclude list.
            is_conv = isinstance(module, nn.Conv2d)
            return is_conv and is_large


def auto_wrap_big_layers(module: nn.Module, fsdp_config: AttrDict):
    """
    Automatically wrap the bigger layer in the module
    """
    with enable_wrap(
        auto_wrap_policy=_BigConvAutoWrapPolicy(fsdp_config.AUTO_WRAP_THRESHOLD),
        wrapper_cls=_FSDP_WRAPPER,
        **fsdp_config,
    ):
        return auto_wrap(module)
