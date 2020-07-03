# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from vissl.models.base_ssl_model import BaseSSLMultiInputOutputModel
from vissl.models.model_helpers import convert_sync_bn


def build_model(model_config, optimizer_config):
    return BaseSSLMultiInputOutputModel(model_config, optimizer_config)


__all__ = ["BaseSSLMultiInputOutputModel", "build_model", "convert_sync_bn"]
