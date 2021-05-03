# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import torch
import torch.nn as nn
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook


class SwAVUpdateQueueScoresHook(ClassyHook):
    """
    Update queue scores, useful with small batches and helps getting
    meaningful gradients.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def on_forward(self, task) -> None:
        """
        If we want to use queue in SwAV training,
        update the queue scores after every forward.
        """
        if not task.config["LOSS"]["name"] == "swav_loss":
            return
        if not task.loss.swav_criterion.use_queue:
            return
        try:
            task.loss.swav_criterion.compute_queue_scores(task.model.heads[0])
        except AttributeError:
            task.loss.swav_criterion.compute_queue_scores(task.model.module.heads[0])


class NormalizePrototypesHook(ClassyHook):
    """
    L2 Normalize the prototypes in swav training. Optional.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Optionally normalize prototypes
        """
        if not task.config["LOSS"]["name"] == "swav_loss":
            return
        if not task.config.LOSS["swav_loss"].normalize_last_layer:
            return
        with torch.no_grad():
            try:
                # This is either single GPU model or a FSDP.
                assert len(task.model.heads) == 1
                for j in range(task.model.heads[0].nmb_heads):
                    module = getattr(task.model.heads[0], "prototypes" + str(j))
                    # Determine the context we need to use. For FSDP, we
                    # need the summon_full_params context, which ensures that
                    # full weights for this layer is all_gathered and after
                    # normalization, the changes are persisted in the local
                    # shards. All ranks do the same normalization, so all
                    # changes should be saved.
                    ctx = contextlib.suppress()
                    if hasattr(module, "summon_full_params"):
                        ctx = module.summon_full_params()
                    with ctx:
                        w = module.weight.data.clone()
                        w = nn.functional.normalize(w, dim=1, p=2)
                        module.weight.copy_(w)
            except AttributeError:
                # This is a DDP wrapped one.
                assert len(task.model.module.heads) == 1
                for j in range(task.model.module.heads[0].nmb_heads):
                    w = getattr(
                        task.model.module.heads[0], "prototypes" + str(j)
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(
                        task.model.module.heads[0], "prototypes" + str(j)
                    ).weight.copy_(w)
