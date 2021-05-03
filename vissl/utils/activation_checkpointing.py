# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module centralizes all activation checkpointing related code.
It is a work-in-progress as we evolve the APIs and eventually put this
in fairscale so that multiple projects can potentially share it.
"""


from typing import Dict, List

import torch.distributed as dist
from torch import nn
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel


def manual_gradient_reduction(model: Module, config_flag: bool) -> bool:
    """
    Return if we should use manual gradient reduction or not.

    We should use manual DDP if config says so and model is wrapped by DDP.
    """
    return config_flag and isinstance(model, DistributedDataParallel)


def manual_sync_params(model: DistributedDataParallel) -> None:
    """
    Manually sync params and buffers for DDP.
    """
    _orig = model.require_forward_param_sync
    model.require_forward_param_sync = True
    model._sync_params()
    model.require_forward_param_sync = _orig


def manual_gradient_all_reduce(model: DistributedDataParallel) -> None:
    """
    Gradient reduction function used after backward is done.
    """
    w = []
    for p in model.parameters():
        if p.grad is not None:
            work = dist.all_reduce(
                p.grad.data, group=model.process_group, async_op=True
            )
            w.append(work)
    for work in w:
        work.wait()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.div_(dist.get_world_size())


def layer_splittable_before(m: Module) -> bool:
    """
    Return if this module can be split in front of it for checkpointing.
    We don't split the relu module.
    """
    return str(m) != "ReLU(inplace=True)"


def checkpoint_trunk(
    feature_blocks: Dict[str, Module],
    unique_out_feat_keys: List[str],
    checkpointing_splits: int,
) -> Dict[str, Module]:
    """
    Checkpoint a list of blocks and return back the split version.
    """
    # If checkpointing, split the model appropriately. The number of features requested
    # can be a limiting factor and override the number of activation chunks requested
    feature_blocks_bucketed = []

    # The features define the splits, first pass
    bucket = []

    for feature_name, feature_block in feature_blocks.items():
        # expand the res2,res3, res4, res5 kind of stages into sub-blocks so that we can
        # checkpoint them.
        if feature_name.startswith("res"):
            for b in feature_block:
                bucket.append(b)
        else:
            bucket.append(feature_block)

        if feature_name in unique_out_feat_keys:
            # Boundary, add to current bucket and move to next
            feature_blocks_bucketed.append([feature_name, bucket])
            bucket = []

    # If there are not enough splits, split again
    split_times = 0
    while len(feature_blocks_bucketed) < checkpointing_splits:
        # Find the biggest block
        lengths = [len(v) for _, v in feature_blocks_bucketed]
        assert max(lengths) > 0, "Something wrong, we shouldn't have an empty list"
        if max(lengths) == 1:
            # Everything is already split.
            break
        if max(lengths) == 2:
            # Find a splittable 2-element element.
            found = False
            for i, (_, v) in enumerate(feature_blocks_bucketed):
                if len(v) == 2 and layer_splittable_before(v[1]):
                    found = True
                    i_max = i
                    break
            if not found:
                # Didn't find good 2-element block, we are done.
                break
        else:
            # TODO: here we assume all >2-element blocks are splittable,
            #       i.e. there is not layer-relu-relu, case. But in general
            #       this is not the case. We can extend in the future.
            i_max = lengths.index(max(lengths))

        # Split the biggest block in two, keep the rest unchanged
        # Avoid inplace-relu.
        n_split_layers = len(feature_blocks_bucketed[i_max][1]) // 2
        biggest_block = feature_blocks_bucketed[i_max]
        if not layer_splittable_before(biggest_block[1][n_split_layers]):
            assert len(biggest_block[1]) > 2
            if n_split_layers == len(biggest_block[1]) - 1:
                n_split_layers -= 1
            else:
                n_split_layers += 1
        assert n_split_layers > 0 and n_split_layers < len(
            biggest_block[1]
        ), "Should never split into an empty list and the rest"

        feature_blocks_bucketed = (
            feature_blocks_bucketed[:i_max]
            + [[f"activation_split_{split_times}", biggest_block[1][:n_split_layers]]]
            + [[biggest_block[0], biggest_block[1][n_split_layers:]]]
            + feature_blocks_bucketed[(i_max + 1) :]
        )
        split_times += 1

    # Replace the model with the bucketed version, checkpoint friendly
    feature_blocks = {
        block[0]: nn.Sequential(*block[1]) for block in feature_blocks_bucketed
    }
    # Make sure we didn't loss anything
    assert len(feature_blocks) == len(feature_blocks_bucketed)
    return feature_blocks
