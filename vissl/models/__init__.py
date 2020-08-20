# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from vissl.models.base_ssl_model import BaseSSLMultiInputOutputModel
from vissl.models.model_helpers import (  # noqa
    convert_sync_bn,
    is_feature_extractor_model,
)


def build_model(model_config, optimizer_config):
    return BaseSSLMultiInputOutputModel(model_config, optimizer_config)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
