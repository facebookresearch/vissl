#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from classy_vision.hooks.classy_hook import ClassyHook


class InitMemoryHook(ClassyHook):
    """
    Initialize the memory banks
    """

    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def on_start(self, task) -> None:
        if not task.config["CRITERION"]["name"] == "deepclusterv2_loss":
            return
        if task.train_phase_idx >= 0:
            return
        task.loss.init_memory(task.dataloaders["train"], task.model)


class ClusterMemoryHook(ClassyHook):
    """
    Cluster the memory banks with distributed k-means
    """

    on_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def on_phase_start(self, task) -> None:
        if not task.config["CRITERION"]["name"] == "deepclusterv2_loss":
            return
        task.loss.cluster_memory()
