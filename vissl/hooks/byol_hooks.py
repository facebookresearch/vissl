# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import logging

import torch
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.models import build_model
from vissl.utils.env import get_machine_local_and_dist_rank

class BYOLHook(ClassyHook):
    """
    BYOL - Bootstrap your own latent: (https://arxiv.org/abs/2006.07733)
    is based on Contrastive learning, this hook
    creates a target network with architecture similar to
    Online network but without the projector head and parameters
    an exponential moving average of the online network's parameters,
    these two networks interact and learn from each other.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop

    @staticmethod
    def cosine_decay(training_iter, max_iters, initial_value)  -> float:
        """
        For a given starting value, this fucntion anneals the learning
        rate.
        """
        training_iter = min(training_iter, max_iters)
        cosine_decay_value = 0.5 * (1 + math.cos(math.pi * training_iter / max_iters))
        return initial_value * cosine_decay_value

    @staticmethod
    def target_ema(training_iter, base_ema, max_iters) -> float:
        """
        Updates Exponential Moving average of the Target Network.
        """
        decay = BYOLHook.cosine_decay(training_iter, max_iters, 1.)
        return 1. - (1. - base_ema) * decay

    def _build_byol_target_network(self, task: tasks.ClassyTask) -> None:
        """
        Creates a "Target Network" which has the same architecture as the
        Online Network but without the projector head and its network parameters
        are a lagging exponential moving average of the online model's parameters.

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

        # TESTED: Target Network and Online network are properly created.
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
        Creates a target network needed for Contrastive learning
        on which BYOL is based. It then updates the target network's
        parameters based on the online network's parameters.
        This function also computer and saves target embeddings,
        which need be can be used for further downstream tasks.
        """
        self._update_momentum_coefficient(task)

        # Update the target model
        if task.loss.target_network is None:
            self._build_byol_target_network(task)
        else:
            self._update_target_network(task)


        # Compute target network embeddings
        batch = task.last_batch.sample['input']
        target_embs = task.loss.target_network(batch)[0]

        # TESTED: Target embeddings are properly saved.

        # Save target embeddings to use them in the loss
        task.loss.target_embs = target_embs
