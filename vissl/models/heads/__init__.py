# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pathlib import Path
from typing import Callable

from classy_vision.generic.registry_utils import import_all_modules


FILE_ROOT = Path(__file__).parent


MODEL_HEADS_REGISTRY = {}
MODEL_HEADS_NAMES = set()


def register_model_head(name: str):
    """Registers Self-Supervision Model Heads.

    This decorator allows VISSL to add custom model heads, even if the
    model head itself is not part of VISSL. To use it, apply this decorator
    to a model head class, like this:

    .. code-block:: python

        @register_model_head('my_model_head_name')
        def my_model_head():
            ...

    To get a model head from a configuration file, see :func:`get_model_head`."""

    def register_model_head_cls(cls: Callable[..., Callable]):
        if name in MODEL_HEADS_REGISTRY:
            raise ValueError("Cannot register duplicate model head ({})".format(name))

        if cls.__name__ in MODEL_HEADS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate model head name ({})".format(
                    cls.__name__
                )
            )
        MODEL_HEADS_REGISTRY[name] = cls
        MODEL_HEADS_NAMES.add(cls.__name__)
        return cls

    return register_model_head_cls


def get_model_head(name: str):
    """
    Given the model head name, construct the head if it's registered
    with VISSL.
    """
    assert name in MODEL_HEADS_REGISTRY, "Unknown model head"
    return MODEL_HEADS_REGISTRY[name]


# automatically import any Python files in the heads/ directory
import_all_modules(FILE_ROOT, "vissl.models.heads")


from vissl.models.heads.linear_eval_mlp import LinearEvalMLP  # isort:skip # noqa
from vissl.models.heads.mlp import MLP  # isort:skip # noqa
from vissl.models.heads.siamese_concat_view import (  # isort:skip  # noqa
    SiameseConcatView,
)
from vissl.models.heads.swav_prototypes_head import (  # isort:skip  # noqa
    SwAVPrototypesHead,
)

__all__ = [
    "get_model_head",
    "LinearEvalMLP",
    "MLP",
    "SiameseConcatView",
    "SwAVPrototypesHead",
]
