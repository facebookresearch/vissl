# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List

import torch
from classy_vision import tasks
from classy_vision.generic.distributed_util import init_distributed_data_parallel_model
from classy_vision.hooks.classy_hook import ClassyHook
from torch import nn
from vissl.models import build_model
from vissl.utils.env import get_machine_local_and_dist_rank


class SwAVMomentumHook(ClassyHook):
    """
    This hook is for the extension of the SwAV loss proposed in paper
    https://arxiv.org/abs/2006.09882 by Caron et al. The loss combines
    the benefits of using the SwAV approach with the momentum encoder
    as used in MoCo.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop

    def __init__(
        self,
        momentum: float,
        momentum_eval_mode_iter_start: int,
        crops_for_assign: List[int],
    ):
        """
        Args:
            momentum (float): for the momentum encoder
            momentum_eval_mode_iter_start (int): from what iteration should the
                            momentum encoder network be in eval mode
            crops_for_assign (List[int]): what crops to use for assignment
        """

        super().__init__()
        self.momentum = momentum
        self.inv_momentum = 1.0 - momentum
        self.crops_for_assign = crops_for_assign
        self.is_distributed = False
        self.momentum_eval_mode_iter_start = momentum_eval_mode_iter_start

    def _build_momentum_network(self, task: tasks.ClassyTask) -> None:
        """
        Create the model replica called the encoder. This will slowly track
        the main model.
        """
        logging.info(
            "Building momentum encoder - rank %s %s", *get_machine_local_and_dist_rank()
        )

        # - same architecture
        task.loss.momentum_encoder = build_model(
            task.config["MODEL"], task.config["OPTIMIZER"]
        )
        task.loss.momentum_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(
            task.loss.momentum_encoder
        )
        task.loss.momentum_encoder.to(torch.device("cuda" if task.use_gpu else "cpu"))

        # Initialize from the model
        if task.loss.checkpoint is None:
            for param_q, param_k in zip(
                task.base_model.parameters(), task.loss.momentum_encoder.parameters()
            ):
                param_k.data.copy_(param_q.data)
            for buff_q, buff_k in zip(
                task.base_model.named_buffers(),
                task.loss.momentum_encoder.named_buffers(),
            ):
                if "running_" not in buff_k[0]:
                    continue
                buff_k[1].data.copy_(buff_q[1].data)
        task.loss.momentum_encoder = init_distributed_data_parallel_model(
            task.loss.momentum_encoder
        )

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

        for buff_q, buff_k in zip(
            task.base_model.named_buffers(), task.loss.momentum_encoder.named_buffers()
        ):
            if "running_" not in buff_k[0]:
                continue
            buff_k[1].data.copy_(buff_q[1].data)

    @torch.no_grad()
    def on_forward(self, task: tasks.ClassyTask) -> None:
        """
        Forward pass with momentum network. We forward momentum encoder
        only on the single resolution crops that are used for assignment
        in the swav loss.
        """

        # Update the momentum encoder
        if task.loss.momentum_encoder is None:
            self._build_momentum_network(task)
        else:
            self._update_momentum_network(task)

        if task.loss.num_iteration >= self.momentum_eval_mode_iter_start:
            task.loss.momentum_encoder.eval()
            if task.loss.num_iteration == self.momentum_eval_mode_iter_start:
                logging.info("Momentum network will be used in eval mode.")
        else:
            task.loss.momentum_encoder.train()

        # Compute momentum features. We do not backpropagate in this codepath
        im_k = [task.last_batch.sample["input"][i] for i in self.crops_for_assign]
        output = task.loss.momentum_encoder(im_k)[0]
        task.loss.momentum_scores = output[1:]
        task.loss.momentum_embeddings = output[0]


class SwAVMomentumNormalizePrototypesHook(ClassyHook):
    """
    L2 Normalize the prototypes in swav training. Optional.
    We normalize the momentum_encoder output prototypes as well
    additionally.
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
        if not task.config["LOSS"]["name"] == "swav_momentum_loss":
            return
        if not task.config.LOSS["swav_momentum_loss"].normalize_last_layer:
            return
        with torch.no_grad():
            try:
                for j in range(task.model.heads[0].nmb_heads):
                    w = getattr(
                        task.model.heads[0], "prototypes" + str(j)
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(task.model.heads[0], "prototypes" + str(j)).weight.copy_(w)
            except AttributeError:
                # TODO (mathildecaron): don't use getattr
                for j in range(task.model.module.heads[0].nmb_heads):
                    w = getattr(
                        task.model.module.heads[0], "prototypes" + str(j)
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(
                        task.model.module.heads[0], "prototypes" + str(j)
                    ).weight.copy_(w)
            try:
                for j in range(task.loss.momentum_encoder.heads[0].nmb_heads):
                    w = getattr(
                        task.loss.momentum_encoder.heads[0], "prototypes" + str(j)
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(
                        task.loss.momentum_encoder.heads[0], "prototypes" + str(j)
                    ).weight.copy_(w)
            except AttributeError:
                for j in range(task.loss.momentum_encoder.module.heads[0].nmb_heads):
                    w = getattr(
                        task.loss.momentum_encoder.module.heads[0],
                        "prototypes" + str(j),
                    ).weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    getattr(
                        task.loss.momentum_encoder.module.heads[0],
                        "prototypes" + str(j),
                    ).weight.copy_(w)
