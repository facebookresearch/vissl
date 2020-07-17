# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# We need to do the imports below so that classy vision registry can run
import vissl.data.ssl_transforms  # NOQA
import vissl.meters  # NOQA
import vissl.models  # NOQA
import vissl.ssl_criterions  # NOQA
import vissl.ssl_hooks  # NOQA
from vissl.ssl_tasks.ssl_task import SelfSupervisionTask


def build_task(config):
    return SelfSupervisionTask.from_config(config)


__all__ = [k for k in globals().keys() if not k.startswith("_")]
