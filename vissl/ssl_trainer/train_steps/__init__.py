#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Here we create all the custom train steps required for SSL model trainings.
"""

from vissl.ssl_trainer.train_steps.standard_train_step import standard_train_step


TRAIN_STEPS = {"standard": standard_train_step}


def get_train_step(train_step_name):
    assert train_step_name in TRAIN_STEPS, "Unknown train step"
    return TRAIN_STEPS[train_step_name]
