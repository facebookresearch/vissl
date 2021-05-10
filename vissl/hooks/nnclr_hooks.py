# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import torch
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.models.heads import MLP
from vissl.utils.env import get_machine_local_and_dist_rank


class NNCLRHook(ClassyHook):
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop

    def __init__(self):
        super().__init__()

    @staticmethod
    def _build_nnclr_predictor(task: tasks.ClassyTask) -> None:
        """
        Create the NNCLR loss predictor.
        """
        # Create the predictor
        logging.info(
            "Building NNCLR predictor - rank %s %s", *get_machine_local_and_dist_rank()
        )

        task.loss.criterion.prediction_head = MLP(
            task.config["MODEL"], **task.loss.criterion.prediction_head_params
        )

        task.loss.criterion.prediction_head.to(task.device)

        # Restore an hypothetical checkpoint
        if task.loss.checkpoint is not None:
            task.loss.load_state_dict(task.loss.checkpoint)

    def on_forward(self, task: tasks.ClassyTask) -> None:
        """
        Compute the predictor embeddings.
        """
        if task.loss.criterion.prediction_head is None:
            self._build_nnclr_predictor(task)

        z = task.last_batch.model_output

        p = task.loss.criterion.prediction_head(z)
        p = torch.nn.functional.normalize(p, dim=1, p=2)

        task.loss.criterion.preds = p
