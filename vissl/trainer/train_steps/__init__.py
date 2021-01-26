# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Here we create all the custom train steps required for SSL model trainings.
"""

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent


TRAIN_STEP_REGISTRY = {}
TRAIN_STEP_NAMES = set()


def register_train_step(name):
    """
    Registers Self-Supervision Train step.

    This decorator allows VISSL to add custom train steps, even if the
    train step itself is not part of VISSL. To use it, apply this decorator
    to a train step function, like this:

    .. code-block:: python

        @register_train_step('my_step_name')
        def my_step_name():
            ...

    To get a train step from a configuration file, see :func:`get_train_step`.
    """

    def register_train_step_fn(func):
        if name in TRAIN_STEP_REGISTRY:
            raise ValueError("Cannot register duplicate train step ({})".format(name))

        if func.__name__ in TRAIN_STEP_NAMES:
            raise ValueError(
                "Cannot register task with duplicate train step name ({})".format(
                    func.__name__
                )
            )
        TRAIN_STEP_REGISTRY[name] = func
        TRAIN_STEP_NAMES.add(func.__name__)
        return func

    return register_train_step_fn


def get_train_step(train_step_name: str):
    """
    Lookup the train_step_name in the train step registry and return.
    If the train step is not implemented, asserts will be thrown and workflow will exit.
    """
    assert train_step_name in TRAIN_STEP_REGISTRY, "Unknown train step"
    return TRAIN_STEP_REGISTRY[train_step_name]


# automatically import any Python files in the train_steps/ directory
import_all_modules(FILE_ROOT, "vissl.trainer.train_steps")
