# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import torch
from classy_vision import tasks
from classy_vision.generic.distributed_util import is_distributed_training_run
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.models import build_model
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.misc import concat_all_gather


class MoCoHook(ClassyHook):
    """
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

    def __init__(self, momentum: float, shuffle_batch: bool = True):
        super().__init__()
        self.momentum = momentum
        self.inv_momentum = 1.0 - momentum
        self.is_distributed = False
        self.shuffle_batch = shuffle_batch

        logging.warning("Batch shuffling: %s", self.shuffle_batch)

    def _build_moco_encoder(self, task: tasks.ClassyTask) -> None:
        """
        Create the model replica called the encoder. This will slowly track
        the main model.
        """
        # Create the encoder, which will slowly track the model
        logging.info(
            "Building MoCo encoder - rank %s %s", *get_machine_local_and_dist_rank()
        )

        # - same architecture
        task.loss.moco_encoder = build_model(
            task.config["MODEL"], task.config["OPTIMIZER"]
        )

        task.loss.moco_encoder.to(task.device)

        # Restore an hypothetical checkpoint, else initialize from the model
        if task.loss.checkpoint is not None:
            task.loss.load_state_dict(task.loss.checkpoint)
        else:
            for param_q, param_k in zip(
                task.base_model.parameters(), task.loss.moco_encoder.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self, task: tasks.ClassyTask) -> None:
        """
        Momentum update of the key encoder:
        Each parameter becomes a weighted average of its old self and the
        newest encoder.
        """

        # Momentum update
        for param_q, param_k in zip(
            task.base_model.parameters(), task.loss.moco_encoder.parameters()
        ):
            param_k.data = (
                param_k.data * self.momentum + param_q.data * self.inv_momentum
            )

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, task: tasks.ClassyTask):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        if task.device.type == "cuda":
            idx_shuffle = torch.randperm(batch_size_all).cuda()
        else:
            idx_shuffle = torch.randperm(batch_size_all)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def on_forward(self, task: tasks.ClassyTask) -> None:
        """
        - Update the momentum encoder.
        - Compute the key reusing the updated moco-encoder. If we use the
          batch shuffling, the perform global shuffling of the batch
          and then run the moco encoder to compute the features.
          We unshuffle the computer features and use the features
          as "key" in computing the moco loss.
        """

        # Update the momentum encoder
        if task.loss.moco_encoder is None:
            self._build_moco_encoder(task)
            self.is_distributed = is_distributed_training_run()
            logging.info("MoCo: Distributed setup, shuffling batches")
        else:
            self._update_momentum_encoder(task)

        # Compute key features. We do not backpropagate in this codepath
        im_k = task.last_batch.sample["data_momentum"][0]

        if self.is_distributed and self.shuffle_batch:
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k, task)

        k = task.loss.moco_encoder(im_k)[0]
        k = torch.nn.functional.normalize(k, dim=1)

        if self.is_distributed and self.shuffle_batch:
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        task.loss.key = k
