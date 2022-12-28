# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


PARAM_GROUP_CONSTRUCTOR_REGISTRY = {}


def register_param_group_constructor(name: str):
    def _register(func):
        if name in PARAM_GROUP_CONSTRUCTOR_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate parameter group constructor ({name})"
            )
        PARAM_GROUP_CONSTRUCTOR_REGISTRY[name] = func
        return func

    return _register


def get_param_group_constructor(model_name: str):
    """
    Lookup the model_name in the model registry and return.
    If the model is not implemented, asserts will be thrown and workflow will exit.
    """
    message = "Unknown parameter group constructor"
    assert model_name in PARAM_GROUP_CONSTRUCTOR_REGISTRY, message
    return PARAM_GROUP_CONSTRUCTOR_REGISTRY[model_name]
