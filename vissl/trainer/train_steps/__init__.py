# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Here we create all the custom train steps required for SSL model trainings.
"""

from vissl.trainer.train_steps.standard_train_step import standard_train_step


TRAIN_STEPS = {"standard": standard_train_step}


def get_train_step(train_step_name):
    assert train_step_name in TRAIN_STEPS, "Unknown train step"
    return TRAIN_STEPS[train_step_name]


__all__ = [k for k in globals().keys() if not k.startswith("_")]
