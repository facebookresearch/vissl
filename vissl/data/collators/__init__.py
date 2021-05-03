# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules
from torch.utils.data.dataloader import default_collate


FILE_ROOT = Path(__file__).parent


COLLATOR_REGISTRY = {}
COLLATOR_NAMES = set()


def register_collator(name):
    """
    Registers Self-Supervision data collators.

    This decorator allows VISSL to add custom data collators, even if the
    collator itself is not part of VISSL. To use it, apply this decorator
    to a collator function, like this:

    .. code-block:: python

        @register_collator('my_collator_name')
        def my_collator_name():
            ...

    To get a collator from a configuration file, see :func:`get_collator`.
    """

    def register_collator_fn(func):
        if name in COLLATOR_REGISTRY:
            raise ValueError("Cannot register duplicate collator ({})".format(name))

        if func.__name__ in COLLATOR_NAMES:
            raise ValueError(
                "Cannot register task with duplicate collator name ({})".format(
                    func.__name__
                )
            )
        COLLATOR_REGISTRY[name] = func
        COLLATOR_NAMES.add(func.__name__)
        return func

    return register_collator_fn


def get_collator(collator_name, collate_params):
    """
    Given the collator name and the collator params, return the collator
    if registered with VISSL. Also supports pytorch default collators.
    """
    if collator_name == "default_collate":
        return default_collate
    else:
        assert collator_name in COLLATOR_REGISTRY, "Unknown collator"
        return partial(COLLATOR_REGISTRY[collator_name], **collate_params)


# automatically import any Python files in the collators/ directory
import_all_modules(FILE_ROOT, "vissl.data.collators")
