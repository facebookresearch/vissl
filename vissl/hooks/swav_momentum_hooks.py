# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import logging

import torch
from torch import nn
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.models import build_model
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.misc import concat_all_gather
from classy_vision.generic.distributed_util import init_distributed_data_parallel_model

class SwAVMomentumHook(ClassyHook):
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop

    def __init__(self, momentum: float, crops_for_assign: List[int]):
        super().__init__()
        self.momentum = momentum
        self.inv_momentum = 1.0 - momentum
        self.crops_for_assign = crops_for_assign
        self.is_distributed = False

    def _build_momentum_network(self, task: tasks.ClassyTask) -> None:
        # Create the encoder, which will slowly track the model
        logging.info(
            "Building momentum encoder - rank %s %s", *get_machine_local_and_dist_rank()
        )

        # - same architecture
        task.loss.momentum_encoder = build_model(
            task.config["MODEL"], task.config["OPTIMIZER"]
        )
        task.loss.momentum_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(task.loss.momentum_encoder)
        task.loss.momentum_encoder.to(torch.device("cuda" if task.use_gpu else "cpu"))

        # Initialize from the model
        if task.loss.checkpoint is None:
            for param_q, param_k in zip(
                task.base_model.parameters(), task.loss.momentum_encoder.parameters()
            ):
                param_k.data.copy_(param_q.data)
        if task.loss.is_distributed:
            task.loss.momentum_encoder = init_distributed_data_parallel_model(task.loss.momentum_encoder)

        # Restore an hypothetical checkpoint
        if task.loss.checkpoint is not None:
            task.loss.load_state_dict(task.loss.checkpoint)

    @torch.no_grad()
    def _update_momentum_network(self, task: tasks.ClassyTask) -> None:
        """
        Momentum update
        Each parameter becomes a weighted average of its old self and the
        newest encoder.
        """

        # Momentum update
        for param_q, param_k in zip(
            task.base_model.parameters(), task.loss.momentum_encoder.parameters()
        ):
            param_k.data = (
                param_k.data * self.momentum + param_q.data * self.inv_momentum
            )

    @torch.no_grad()
    def on_forward(self, task: tasks.ClassyTask) -> None:
        """
        Forward pass with momentum network
        """

        # Update the momentum encoder
        if task.loss.momentum_encoder is None:
            self._build_momentum_network(task)
        else:
            self._update_momentum_network(task)

        # Compute momentum features. We do not backpropagate in this codepath
        im_k = [task.last_batch.sample["input"][i] for i in self.crops_for_assign]
        output = task.loss.momentum_encoder(im_k)[0]
        task.loss.momentum_scores = output[1:]
        task.loss.momentum_embeddings = output[0]


class SwAVMomentumNormalizePrototypesHook(ClassyHook):
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
        if not task.config["LOSS"]["name"] == "swav_momentum_loss":
            return
        if not task.config.LOSS["swav_momentum_loss"].normalize_last_layer:
            return
        with torch.no_grad():
            if not task.loss.is_distributed:
                assert len(task.model.heads) == 1
                for j in range(task.model.heads[0].nmb_heads):
                    w = getattr(
                        task.model.heads[0], "prototypes" + str(j)
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(task.model.heads[0], "prototypes" + str(j)).weight.copy_(w)
                    w = getattr(
                        task.loss.momentum_encoder.heads[0], "prototypes" + str(j)
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(task.loss.momentum_encoder.heads[0], "prototypes" + str(j)).weight.copy_(w)
            else:
                assert len(task.model.module.heads) == 1
                for j in range(task.model.module.heads[0].nmb_heads):
                    w = getattr(
                        task.model.module.heads[0], "prototypes" + str(j)
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(
                        task.model.module.heads[0], "prototypes" + str(j)
                    ).weight.copy_(w)
                    w = getattr(
                        task.loss.momentum_encoder.module.heads[0], "prototypes" + str(j)
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(
                        task.loss.momentum_encoder.module.heads[0], "prototypes" + str(j)
                    ).weight.copy_(w)
