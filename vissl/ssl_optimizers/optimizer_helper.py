# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Any, List

import torch.nn as nn
from vissl.utils.misc import is_apex_available


_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

_BN_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,  # pytorch SyncBN
)

if is_apex_available():
    import apex

    _BN_TYPES += (apex.parallel.SyncBatchNorm,)


def _get_bn_optimizer_params(
    module, regularized_params, unregularized_params, optimizer_config
):
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
    # Keep only the trainable params
    return list(filter(lambda x: x.requires_grad, param_list))


def get_optimizer_regularized_params(model, model_config, optimizer_config):
    """
    Go through all the layers, sort out which parameters should be regularized

    Returns:
        Dict -- Regularized and un-regularized params
    """
    regularized_params, unregularized_params = [], []
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, _CONV_TYPES):
            regularized_params.append(module.weight)
            if module.bias is not None:
                if optimizer_config["regularize_bias"]:
                    regularized_params.append(module.bias)
                else:
                    unregularized_params.append(module.bias)
        elif isinstance(module, _BN_TYPES):
            (regularized_params, unregularized_params) = _get_bn_optimizer_params(
                module, regularized_params, unregularized_params, optimizer_config
            )
        elif len(list(module.children())) >= 0:
            # for any other layers not bn_types, conv_types or nn.Linear, if
            # the layers are the leaf nodes and have parameters, we regularize
            # them. Similarly, if non-leaf nodes but have parameters, regularize
            # them (set recurse=False)
            for params in module.parameters(recurse=False):
                regularized_params.append(params)

    # set the requires_grad to False
    non_trainable_params = []
    for name, param in model.named_parameters():
        if name in model_config.NON_TRAINABLE_PARAMS:
            param.requires_grad = False
            non_trainable_params.append(param)

    trainable_params = _filter_trainable(model.parameters())
    regularized_params = _filter_trainable(regularized_params)
    unregularized_params = _filter_trainable(unregularized_params)
    logging.info(
        f"Traininable params: {len(trainable_params)}, "
        f"Non-Traininable params: {len(non_trainable_params)}, "
        f"Regularized Parameters: {len(regularized_params)}, "
        f"Unregularized Parameters {len(unregularized_params)}"
    )

    return {
        "regularized_params": regularized_params,
        "unregularized_params": unregularized_params,
    }
