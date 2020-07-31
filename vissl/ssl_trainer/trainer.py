# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import gc
import logging
import os
import socket
import time
from typing import Tuple

import numpy as np
import torch
from classy_vision.generic.distributed_util import (
    barrier,
    is_primary,
    set_cpu_device,
    set_cuda_device_index,
)
from classy_vision.tasks import ClassyTask
from classy_vision.trainer import ClassyTrainer
from vissl.data import print_sampler_config
from vissl.ssl_hooks import SSLClassyHookFunctions
from vissl.ssl_trainer.train_steps import get_train_step
from vissl.utils.env import get_machine_local_and_dist_rank


class DistributedSelfSupervisionTrainer(ClassyTrainer):
    def __init__(self, dist_run_id):
        self.dist_run_id = dist_run_id

    def _setup_distributed(self, cfg, use_gpu):

        # we overwrite the distributed trainer setup here with our config options
        distributed_world_size = int(os.environ["WORLD_SIZE"])
        assert distributed_world_size % cfg.DISTRIBUTED.NUM_NODES == 0
        init_method = f"{cfg.DISTRIBUTED.INIT_METHOD}://{self.dist_run_id}"
        logging.info(
            f"Using Distributed init method: {init_method}, "
            f"world_size: {distributed_world_size}, rank: {self.distributed_rank}"
        )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=cfg.DISTRIBUTED.BACKEND,
                init_method=init_method,
                world_size=distributed_world_size,
                rank=self.distributed_rank,
            )
        else:
            logging.warning(
                "Torch distributed has already been initialized, \
                reusing existing configuration"
            )

        logging.info(
            "| initialized host {} as rank {} ({})".format(
                socket.gethostname(),
                self.distributed_rank,
                torch.distributed.get_rank(),
            )
        )
        if use_gpu:
            set_cuda_device_index(self.local_rank)
        else:
            set_cpu_device()

    def train(self, cfg, task: ClassyTask):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()

        self._setup_distributed(cfg, task.use_gpu)

        train_step_fn = get_train_step(cfg["TRAINER"]["TRAIN_STEP_NAME"])
        task.prepare(device=cfg.MACHINE.DEVICE, pin_memory=cfg.DATA.PIN_MEMORY)
        task.init_distributed_data_parallel_model()

        # Find what phase, train_phase_idx, local_iteration_num we are starting from.
        # Recover it from the checkpoint (if available)
        task, phase_idx, iteration_num = self._update_training_state(cfg, task)

        # Good to go, (re) start training
        task.run_hooks(SSLClassyHookFunctions.on_start.name)

        if is_primary():
            logging.info("Model is:\n {}".format(task.model))
            logging.info("Loss is: {}".format(task.loss))
        logging.info("Starting training....")

        while phase_idx + 1 < len(task.phases):
            self._advance_phase(task)  # advances task.phase_idx
            phase_idx += 1
            iteration_num += 1
            task.local_iteration_num = iteration_num  # iteration_num=0 at this step
            task.run_hooks(SSLClassyHookFunctions.on_phase_start.name)
            while True:
                try:
                    task = train_step_fn(task)
                    iteration_num += 1
                    task.local_iteration_num = iteration_num
                    task.run_hooks(SSLClassyHookFunctions.on_step.name)
                except StopIteration:
                    break
            for meter in task.meters:
                meter.sync_state()
            logging.info("Meters synced")
            barrier()
            task.run_hooks(SSLClassyHookFunctions.on_phase_end.name)

        task.run_hooks(SSLClassyHookFunctions.on_end.name)
        if hasattr(task, "data_iterator"):
            del task.data_iterator
            gc.collect()
        if hasattr(task, "dataloaders"):
            del task.dataloaders
            gc.collect()

    @staticmethod
    def _update_training_state(cfg, task: ClassyTask) -> Tuple[ClassyTask, int, int]:
        """If a checkpoint is present, recover the current training status.
        If not initialize everything properly

        Arguments:
            task {ClassyTask} -- [description]

        Returns:
            task {ClassyTask} -- updated task
            phase_idx {int} -- phase index
            iteration_num {int} -- iteration number
        """

        phase_idx, iteration_num = -1, -1
        task.num_phases = len(task.phases)
        task.num_epochs = task.num_train_phases
        task.max_iteration = task.num_epochs * len(task.dataloaders["train"])

        if task.checkpoint is not None:
            phase_idx = task.checkpoint["phase_idx"]
            task.train_phase_idx = task.checkpoint["train_phase_idx"]
            task.local_iteration_num = task.checkpoint["iteration_num"]
            task.iteration = task.checkpoint["iteration"]
        else:
            task.iteration = 0
            task.local_iteration_num = iteration_num
        num_iter_in_epoch = len(task.dataloaders["train"])
        num_samples = task.dataloaders["train"].dataset.num_samples()
        task.start_time = time.time()
        task.batch_time = []
        task.metrics = {}
        logging.info(
            f"Training {task.num_epochs} epochs. One epoch = {num_iter_in_epoch} iterations"
        )
        logging.info(f"Total {task.max_iteration} iterations for training")
        logging.info(f"Total {num_samples} samples in one epoch")
        return task, phase_idx, task.local_iteration_num

    def _advance_phase(self, task):
        # reset the meters at the beginning of the epoch
        for meter in task.meters:
            meter.reset()

        # reset the loss history for this epoch
        task.losses = []

        # advance the epoch num to be current
        task.phase_idx += 1
        phase = task.phases[task.phase_idx]
        task.train = True if phase["train"] else False
        if task.train:
            task.train_phase_idx += 1

        # get the data iterator - delete the iterator at the beginning explicitly
        # so that all dataloader processes are cleaned up
        phase_type = "train" if phase["train"] else "test"
        if hasattr(task.dataloaders[phase_type], "sampler"):
            # (Re-)Shuffle data: set epoch of distributed sampler.
            if hasattr(task.dataloaders[phase_type].sampler, "set_epoch"):
                # task.phase_idx is current running phase id
                task.dataloaders[phase_type].sampler.set_epoch(task.phase_idx)
                logging.info(
                    f"rank: {self.distributed_rank} "
                    f"Epoch set for the sampler: {task.phase_idx}"
                )
            # resume from the iteration if valid
            if hasattr(task.dataloaders[phase_type].sampler, "set_start_iter"):
                if (
                    task.checkpoint is not None
                    and task.checkpoint["iteration"] > 0
                    and task.checkpoint["train_phase_idx"] == (task.train_phase_idx - 1)
                ):
                    num_iters_in_epochs = len(task.dataloaders[phase_type])
                    num_epochs = task.checkpoint["train_phase_idx"] + 1
                    num_train_iters_done = num_epochs * num_iters_in_epochs
                    start_iter = task.checkpoint["iteration"] - num_train_iters_done
                else:
                    start_iter = 0
                task.dataloaders[phase_type].sampler.set_start_iter(start_iter)
                # we recreate the iterator since we adjust things in the sampler
                task.recreate_data_iterator(phase_type)
            print_sampler_config(task.dataloaders[phase_type].sampler)
        task.recreate_data_iterator(phase_type)

        # set the model to train or eval depending on what phase we are in
        task.model.train(phase["train"])

        local_rank, _ = get_machine_local_and_dist_rank()
        logging.info(f"Phase advanced. Rank: {local_rank}")

    def extract(self, cfg, task: ClassyTask):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()

        self._setup_distributed(cfg, task.use_gpu)

        # support feature extraction on gpu only.
        assert task.use_gpu, "Set MACHINE.DEVICE = gpu"
        task.prepare_extraction(
            device=cfg.MACHINE.DEVICE,
            pin_memory=cfg.DATA.PIN_MEMORY,
            use_gpu=task.use_gpu,
        )

        task.init_distributed_data_parallel_model()

        if is_primary():
            logging.info("Model is:\n {}".format(task.model))
        features = {}
        for split in task.available_splits:
            logging.info(f"Extracting features for partition: {split.lower()}")
            task.data_iterator = iter(task.dataloaders[split.lower()])
            features[split.lower()] = self._get_split_features(cfg, task, task.model)
            logging.info(f"Done getting features for partition: {split.lower()}")

        if hasattr(task, "data_iterator"):
            del task.data_iterator
            gc.collect()
        if hasattr(task, "dataloaders"):
            del task.dataloaders
            gc.collect()
        return features

    def _flatten_features_list(self, features):
        assert isinstance(features, list), "features must be of type list"
        is_nested = isinstance(features[0], list)
        if is_nested:
            flat_features_list = [item for sublist in features for item in sublist]
            return flat_features_list
        return features

    def _get_split_features(self, cfg, task, model):
        model.eval()
        # if user doesn't specify the features to evaluate, we get the full model
        # output and freeze head/trunk both as caution.
        # TODO (prigoyal): pass "heads" to the EVAL_FEATURES
        if len(cfg.MODEL.EVAL_FEATURES) == 0:
            feat_layers = ["heads"]
        else:
            feat_layers = cfg.MODEL.EVAL_FEATURES
        out_features, out_targets = {}, {}
        for layer in feat_layers:
            out_features[layer], out_targets[layer] = {}, {}

        while True:
            try:
                sample = next(task.data_iterator)
                assert isinstance(sample, dict)
                assert "data_idx" in sample, "Indices not passed"
                input_sample = {
                    "input": torch.cat(sample["data"]).cuda(non_blocking=True),
                    "target": torch.cat(sample["label"]).numpy(),
                    "inds": torch.cat(sample["data_idx"]).numpy(),
                }
                with torch.no_grad():
                    features = model(input_sample["input"])
                    flat_features_list = self._flatten_features_list(features)
                    num_images = input_sample["inds"].shape[0]
                    for num, layer in enumerate(feat_layers):
                        feature = flat_features_list[num].cpu().numpy()
                        targets = input_sample["target"]
                        for idx in range(num_images):
                            index = input_sample["inds"][idx]
                            if not (index in out_features[layer]):
                                out_targets[layer][index] = targets[idx].reshape(-1)
                                out_features[layer][index] = feature[idx]
            except StopIteration:
                break
        barrier()

        output = {}
        for layer in feat_layers:
            out_features[layer] = dict(sorted(out_features[layer].items()))
            out_targets[layer] = dict(sorted(out_targets[layer].items()))
            feats = np.array(list(out_features[layer].values()))
            N = feats.shape[0]
            output[layer] = {
                "features": feats.reshape(N, -1),
                "targets": np.array(list(out_targets[layer].values())),
                "inds": np.array(list(out_features[layer].keys())),
            }
        return output
