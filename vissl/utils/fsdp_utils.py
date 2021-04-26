# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch.distributed as dist
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP


def fsdp_recursive_reset_lazy_init(fsdp_module: FSDP):
    """
    Before the first forward pass, an FSDP module might have been initialized
    for instance by calling load_state_dict or load_local_state_dict to
    reload a previous training checkpoint.

    This function will recursively walk though the sub-FSDP modules and
    call _reset_lazy_init on each of them.
    """
    to_visit = list(fsdp_module.named_modules())
    while to_visit:
        name, module = to_visit.pop()
        if isinstance(module, FSDP) and module._is_root is not None:
            module._reset_lazy_init()
        for child_name, child in module.named_modules():
            if child_name:
                to_visit.append((name + "." + child_name, child))


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


def fsdp_wrapper(module, **kwargs):
    """
    Customer FSDP wrapper, adding the missing options
    """
    fsdp_config = dict(**kwargs)
    fsdp_config["process_group"] = get_global_group()
    return FSDP(module, **fsdp_config)
