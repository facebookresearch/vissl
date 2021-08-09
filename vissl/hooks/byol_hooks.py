# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import logging

import torch
from classy_vision import tasks
from classy_vision.generic.distributed_util import is_distributed_training_run
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.models import build_model
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.misc import concat_all_gather




class BYOLHook(ClassyHook):
    """
    TODO: Update description

    This hook corresponds to the loss proposed in the "Momentum Contrast
    for Unsupervised Visual Representation Learning" paper, from Kaiming He et al.
    See http://arxiv.org/abs/1911.05722 for details
    and https://github.com/facebookresearch/moco for a reference implementation,
    reused here.

    Called after every forward pass to update the momentum encoder. At the beginning
    of training i.e. after 1st forward call, the encoder is contructed and updated.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop

    def __init__(self, base_momentum: float, shuffle_batch: bool = True):
        super().__init__()
        self.base_momentum = base_momentum
        self.is_distributed = False

        self.momentum = None
        self.inv_momentum = None
        self.total_iters = None

    @staticmethod
    def cosine_decay(training_iter, max_iters, initial_value):
        # TODO: Why do we need this min statement?
        training_iter = min(training_iter, max_iters)
        cosine_decay_value = 0.5 * (1 + math.cos(math.pi * training_iter / max_iters))
        return initial_value * cosine_decay_value

    @staticmethod
    def target_ema(training_iter, base_ema, max_iters):
        decay = BYOLHook.cosine_decay(training_iter, max_iters, 1.)
        return 1. - (1. - base_ema) * decay

    def _build_byol_target_network(self, task: tasks.ClassyTask) -> None:
        """
        Create the model replica called the target. This will slowly track
        the online model.
        """
        # Create the encoder, which will slowly track the model
        logging.info(
            "BYOL: Building BYOL target network - rank %s %s", *get_machine_local_and_dist_rank()
        )

        # Target model has the same architecture, *without* the projector head.
        target_model_config = task.config['MODEL']
        target_model_config['HEAD']['PARAMS'] = target_model_config['HEAD']['PARAMS'][0:1]
        task.loss.target_network = build_model(
            target_model_config, task.config["OPTIMIZER"]
        )

        # TESTED: Target Network and other network is properly created.
        # TODO: Check SyncBatchNorm settings (low prior)

        task.loss.target_network.to(task.device)

        # Restore an hypothetical checkpoint, else copy the model parameters from the
        # online network.
        if task.loss.checkpoint is not None:
            task.loss.load_state_dict(task.loss.checkpoint)
        else:
            logging.info("BYOL: Copying and freezing model parameters from online to target network")
            for param_q, param_k in zip(
                task.base_model.parameters(), task.loss.target_network.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        # TESTED: Network is properly copied.
        logging.info("BYOL target network:")
        logging.info(task.loss.target_network)

    def _update_momentum_coefficient(self, task: tasks.ClassyTask) -> None:
        """
        Update the momentum coefficient based on the task config.
        """
        if self.total_iters is None:
            self.total_iters = task.max_iteration
            logging.info(f"{self.total_iters} total iters")

        training_iteration = task.iteration

        self.momentum = self.target_ema(training_iteration, self.base_momentum, self.total_iters)

    @torch.no_grad()
    def _update_target_network(self, task: tasks.ClassyTask) -> None:
        """
        Momentum update of the key encoder:
        Each parameter becomes a weighted average of its old self and the
        newest encoder.
        """
        # Momentum update
        for online_params, target_params in zip(
            task.base_model.parameters(), task.loss.target_network.parameters()
        ):
            target_params.data = (
                target_params.data * self.momentum + online_params.data * (1. - self.momentum)
            )
            # TESTED: PROPER Momentum update.


    @torch.no_grad()
    def on_forward(self, task: tasks.ClassyTask) -> None:
        """
        - Update the target model.
        - Compute the key reusing the updated moco-encoder. If we use the
          batch shuffling, the perform global shuffling of the batch
          and then run the moco encoder to compute the features.
          We unshuffle the computer features and use the features
          as "key" in computing the moco loss.
        """
        self._update_momentum_coefficient(task)

        # Update the target model
        if task.loss.target_network is None:
            self._build_byol_target_network(task)
            # TODO: Do we need this or this is an artifact from moco_hooks.py?
            self.is_distributed = is_distributed_training_run()
        else:
            self._update_target_network(task)


        # Compute target network embeddings
        batch = task.last_batch.sample['input']
        target_embs = task.loss.target_network(batch)[0]

        # TESTED: Target embeddings are properly saved.

        # Save target embeddings to use them in the loss
        task.loss.target_embs = target_embs
