# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pathlib import Path
from typing import Callable

from classy_vision.generic.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent


MODEL_TRUNKS_REGISTRY = {}
MODEL_TRUNKS_NAMES = set()


def register_model_trunk(name: str):
    """Registers Self-Supervision Model Trunks.

    This decorator allows VISSL to add custom model trunk, even if the
    model trunk itself is not part of VISSL. To use it, apply this decorator
    to a model trunk class, like this:

    .. code-block:: python

        @register_model_trunk('my_model_trunk_name')
        def my_model_trunk():
            ...

    To get a model trunk from a configuration file, see :func:`get_model_trunk`."""

    def register_model_trunk_cls(cls: Callable[..., Callable]):
        if name in MODEL_TRUNKS_REGISTRY:
            raise ValueError("Cannot register duplicate model trunk ({})".format(name))

        if cls.__name__ in MODEL_TRUNKS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate model trunk name ({})".format(
                    cls.__name__
                )
            )
        MODEL_TRUNKS_REGISTRY[name] = cls
        MODEL_TRUNKS_NAMES.add(cls.__name__)
        return cls

    return register_model_trunk_cls


def get_model_trunk(name: str):
    """
    Given the model trunk name, construct the trunk if it's registered
    with VISSL.
    """
    assert name in MODEL_TRUNKS_REGISTRY, "Unknown model trunk"
    return MODEL_TRUNKS_REGISTRY[name]


# automatically import any Python files in the trunks/ directory
import_all_modules(FILE_ROOT, "vissl.models.trunks")
