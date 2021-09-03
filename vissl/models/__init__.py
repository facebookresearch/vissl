# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Here we create all the custom models required for SSL.
"""

from pathlib import Path

from classy_vision.generic.registry_utils import import_all_modules
from classy_vision.models import MODEL_REGISTRY


def get_model(model_name: str):
    """
    Lookup the model_name in the model registry and return.
    If the model is not implemented, asserts will be thrown and workflow will exit.
    """
    assert model_name in MODEL_REGISTRY, "Unknown model"
    return MODEL_REGISTRY[model_name]


def build_model(model_config, optimizer_config):
    """
    Given the model config and the optimizer config, construct the model.
    The returned model is not copied to gpu yet (if using gpu) and neither
    wrapped with DDP yet. This is done later train_task.py .prepare()
    """
    model_name = model_config.NAME
    model_cls = get_model(model_name)
    return model_cls(model_config, optimizer_config)


from vissl.models.model_helpers import (  # noqa
    convert_sync_bn,
    is_feature_extractor_model,
)


# automatically import any Python files in the models/ directory
FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "vissl.models")
