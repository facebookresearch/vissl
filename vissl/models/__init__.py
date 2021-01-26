# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from vissl.models.base_ssl_model import BaseSSLMultiInputOutputModel
from vissl.models.model_helpers import (  # noqa
    convert_sync_bn,
    is_feature_extractor_model,
)


def build_model(model_config, optimizer_config):
    """
    Given the model config and the optimizer config, construct the model.
    The returned model is not copied to gpu yet (if using gpu) and neither
    wrapped with DDP yet. This is done later train_task.py .prepare()
    """
    return BaseSSLMultiInputOutputModel(model_config, optimizer_config)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
