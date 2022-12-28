# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List

import torch.nn as nn
from fvcore.common.param_scheduler import ParamScheduler
from vissl.config import AttrDict
from vissl.optimizers.param_groups.registry import register_param_group_constructor
from vissl.utils.misc import is_apex_available


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


def _get_norm_optimizer_params(
    module: nn.Module,
    regularized_params: List[nn.Parameter],
    unregularized_params: List[nn.Parameter],
    optimizer_config: AttrDict,
):
    """
    Given the (Sync)BatchNorm module in the model, we separate the module params
    into regularized or non-regularized (weight_decay=0).
    """
    # this is called by get_optimizer_params for BN specific layer only
    if module.weight is not None:
        if optimizer_config["regularize_bn"]:
            regularized_params.append(module.weight)
        else:
            unregularized_params.append(module.weight)
    if module.bias is not None:
        if optimizer_config["regularize_bn"] and optimizer_config["regularize_bias"]:
            regularized_params.append(module.bias)
        else:
            unregularized_params.append(module.bias)
    return regularized_params, unregularized_params


def _filter_trainable(param_list: List[Any]) -> List[Any]:
    """
    Keep on the trainable parameters of the model.
    """
    return [x for x in param_list if x.requires_grad]


def _assign_regularized_params(
    regularized_param_list=None,
    unregularized_param_list=None,
    parameters_to_unregularize=None,
):
    """
    Takes a list parameters_to_unregularize (a list of parameters to ensure are
    not regularized) and compares it to regularized_param_list, a list of
    regularized parameters. Any parameters in parameters_to_unregularize that
    are present in regularized_param_list are removed from
    regularized_param_list. Will also check against an optional
    unregularized_param_list (pre-existing list of parameters not to regularize)
    and remove any items from parameters_to_unregularize that are in
    unregularized_param_list. Used for when we have parameters that we don't
    want to regularize (e.g. the class token and position embeddings for the
    vision transformer). See config.OPTIMIZER.non_regularized_params. Needs
    to be called separately for head, trunk, and remaining params.
    """
    indices_to_remove_from_regularized = []
    indices_to_remove_from_new_unregularized = []

    # Iterate through new parameters to unregularize
    for unreg_param_ind, new_unreg_param in enumerate(parameters_to_unregularize):
        # Iterate through list of regularized parameters
        for reg_param_ind, reg_param in enumerate(regularized_param_list):
            # Note any matches
            if reg_param is new_unreg_param:
                indices_to_remove_from_regularized.append(reg_param_ind)
        if unregularized_param_list:
            # Iterate through pre-existing list of unregularized parameters
            for unreg_param in unregularized_param_list:
                # Note any matches
                if unreg_param is new_unreg_param:
                    indices_to_remove_from_new_unregularized.append(unreg_param_ind)

    # Iterate through indices to remove from list regularized params and
    # remove them
    indices_to_remove_from_regularized.sort(reverse=True)
    for i in indices_to_remove_from_regularized:
        del regularized_param_list[i]
    if unregularized_param_list:
        indices_to_remove_from_new_unregularized.sort(reverse=True)
        # Iterate through indices to remove from new list of unregularized
        # parameters
        for i in indices_to_remove_from_new_unregularized:
            del parameters_to_unregularize[i]
    return parameters_to_unregularize, regularized_param_list, unregularized_param_list


@register_param_group_constructor("default")
def get_default_optimizer_param_groups(
    model: nn.Module,
    model_config: AttrDict,
    optimizer_config: AttrDict,
    optimizer_schedulers: Dict[str, ParamScheduler],
):
    """
    Go through all the layers, sort out which parameters should be regularized,
    unregularized and optimization settings for the head/trunk. We filter
    the trainable params only and add them to the param_groups.

    Returns:
        param_groups (List[Dict]): [
            {
                "params": trunk_regularized_params, "lr": lr_value,
                "weight_decay": wd_value,
            },
            {
                "params": trunk_unregularized_params, "lr": lr_value,
                "weight_decay": 0.0,
            },
            {
                "params": head_regularized_params, "lr": head_lr_value,
                "weight_decay": head_weight_decay,
            },
            {
                "params": head_unregularized_params, "lr": head_lr_value,
                "weight_decay": 0.0,
            },
            {
                "params": remaining_regularized_params, "lr": lr_value
            }
        ]
    """
    if "weight_decay" in optimizer_schedulers:
        weight_decay_main_config = optimizer_schedulers["weight_decay"]
    else:
        weight_decay_main_config = optimizer_config.weight_decay

    if "weight_decay_head" in optimizer_schedulers:
        weight_decay_head_main_config = optimizer_schedulers["weight_decay_head"]
    else:
        weight_decay_head_main_config = (
            optimizer_config.head_optimizer_params.weight_decay
        )

    if optimizer_config.construct_single_param_group_only:
        # If single param_group is asked, we just use the parameters
        # returned from model.parameters(). This is useful in FSDP
        # param flattening mode.
        return [
            {
                "params": list(model.parameters()),
                "lr": optimizer_schedulers["lr"],
                "weight_decay": weight_decay_main_config,
            }
        ]

    # if the different LR, weight decay value for head is not specified, we use the
    # same LR/wd as trunk.
    if not optimizer_config.head_optimizer_params.use_different_lr:
        assert "lr_head" in optimizer_schedulers

    # we create 6 params groups:
    # - trunk vs head vs other
    # - regularized or un-regularized
    # Un-regularized can contain BN layers.
    trunk_regularized_params, trunk_unregularized_params = [], []
    head_regularized_params, head_unregularized_params = [], []
    regularized_params, unregularized_params = [], []

    for name, module in model.named_modules():

        # head, Linear/Conv layer
        if "head" in name and (
            isinstance(module, nn.Linear) or isinstance(module, _CONV_TYPES)
        ):
            # weight normalized linear layers, used in swav_prototypes_head for example,
            # have "weight_g" and "weight_v" parameters in place of "weight"
            if hasattr(module, "weight_g"):
                head_regularized_params.append(module.weight_g)
                head_regularized_params.append(module.weight_v)
            else:
                head_regularized_params.append(module.weight)
            if module.bias is not None:
                if optimizer_config["regularize_bias"]:
                    head_regularized_params.append(module.bias)
                else:
                    head_unregularized_params.append(module.bias)

        # head, BN/LN layer
        elif "head" in name and isinstance(module, _NORM_TYPES):
            (
                head_regularized_params,
                head_unregularized_params,
            ) = _get_norm_optimizer_params(
                module,
                head_regularized_params,
                head_unregularized_params,
                optimizer_config,
            )

        # trunk, Linear/Conv
        elif isinstance(module, nn.Linear) or isinstance(module, _CONV_TYPES):
            if hasattr(module, "weight_g"):  # weight_norm linear layers
                trunk_regularized_params.append(module.weight_g)
                trunk_regularized_params.append(module.weight_v)
            else:
                trunk_regularized_params.append(module.weight)
            if module.bias is not None:
                if optimizer_config["regularize_bias"]:
                    trunk_regularized_params.append(module.bias)
                else:
                    trunk_unregularized_params.append(module.bias)

        # trunk, BN/LN layer
        elif isinstance(module, _NORM_TYPES):
            (
                trunk_regularized_params,
                trunk_unregularized_params,
            ) = _get_norm_optimizer_params(
                module,
                trunk_regularized_params,
                trunk_unregularized_params,
                optimizer_config,
            )
        elif len(list(module.children())) >= 0:
            # for any other layers not bn_types, conv_types or nn.Linear, if
            # the layers are the leaf nodes and have parameters, we regularize
            # them. Similarly, if non-leaf nodes but have parameters, regularize
            # them (set recurse=False)
            for params in module.parameters(recurse=False):
                regularized_params.append(params)

    # Collect user-specified non-regularized params and remove them for the
    # lists of regularized params, and check they're not already on the lists
    # of unregularized params
    if optimizer_config.non_regularized_parameters:
        non_reg_param_names = optimizer_config.non_regularized_parameters
        for name, param in model.named_parameters():
            hits = [p for p in non_reg_param_names if p in name]
            if any(hits):
                unregularized_params.append(param)
        # Call for trunk params
        (
            non_reg_params,
            trunk_regularized_params,
            trunk_unregularized_params,
        ) = _assign_regularized_params(
            parameters_to_unregularize=unregularized_params,
            regularized_param_list=trunk_regularized_params,
            unregularized_param_list=trunk_unregularized_params,
        )
        # Call for head params
        (
            non_reg_params,
            head_regularized_params,
            head_unregularized_params,
        ) = _assign_regularized_params(
            parameters_to_unregularize=unregularized_params,
            regularized_param_list=head_regularized_params,
            unregularized_param_list=head_unregularized_params,
        )
        # Call for remaining params
        non_reg_params, regularized_params, _ = _assign_regularized_params(
            parameters_to_unregularize=unregularized_params,
            regularized_param_list=regularized_params,
        )

    # for non-trainable params, set the requires_grad to False
    non_trainable_params = []
    for name, param in model.named_parameters():
        if name in model_config.NON_TRAINABLE_PARAMS:
            param.requires_grad = False
            non_trainable_params.append(param)

    trainable_params = _filter_trainable(model.parameters())
    trunk_regularized_params = _filter_trainable(trunk_regularized_params)
    trunk_unregularized_params = _filter_trainable(trunk_unregularized_params)
    head_regularized_params = _filter_trainable(head_regularized_params)
    head_unregularized_params = _filter_trainable(head_unregularized_params)
    regularized_params = _filter_trainable(regularized_params)
    logging.info(
        f"\nTrainable params: {len(trainable_params)}, \n"
        f"Non-Trainable params: {len(non_trainable_params)}, \n"
        f"Trunk Regularized Parameters: {len(trunk_regularized_params)}, \n"
        f"Trunk Unregularized Parameters {len(trunk_unregularized_params)}, \n"
        f"Head Regularized Parameters: {len(head_regularized_params)}, \n"
        f"Head Unregularized Parameters: {len(head_unregularized_params)} \n"
        f"Remaining Regularized Parameters: {len(regularized_params)} \n"
        f"Remaining Unregularized Parameters: {len(unregularized_params)}"
    )

    param_groups = [
        {
            "params": trunk_regularized_params,
            "lr": optimizer_schedulers["lr"],
            "weight_decay": weight_decay_main_config,
        },
        {
            "params": trunk_unregularized_params,
            "lr": optimizer_schedulers["lr"],
            "weight_decay": 0.0,
        },
        {
            "params": head_regularized_params,
            "lr": optimizer_schedulers["lr_head"],
            "weight_decay": weight_decay_head_main_config,
        },
        {
            "params": head_unregularized_params,
            "lr": optimizer_schedulers["lr_head"],
            "weight_decay": 0.0,
        },
    ]
    if len(regularized_params) > 0:
        param_groups.append(
            {
                "params": regularized_params,
                "lr": optimizer_schedulers["lr"],
                "weight_decay": weight_decay_main_config,
            }
        )
    if len(unregularized_params) > 0:
        param_groups.append(
            {
                "params": unregularized_params,
                "lr": optimizer_schedulers["lr"],
                "weight_decay": 0.0,
            }
        )

    return param_groups
