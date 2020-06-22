#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# We need to do the imports below so that classy vision registry can run
import vissl.dataset.ssl_transforms  # NOQA
import vissl.meters  # NOQA
import vissl.models  # NOQA
import vissl.ssl_criterions  # NOQA
import vissl.ssl_hooks  # NOQA
from vissl.tasks.ssl_task import SelfSupervisionTask


def build_task(config):
    return SelfSupervisionTask.from_config(config)


__all__ = ["SelfSupervisionTask"]
