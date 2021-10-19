# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import os
import socket
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    barrier,
    is_primary,
    set_cpu_device,
    set_cuda_device_index,
)
from classy_vision.generic.util import copy_model_to_gpu
from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.tasks import TASK_REGISTRY, ClassyTask
from vissl.config import AttrDict
from vissl.hooks import SSLClassyHookFunctions
from vissl.models.model_helpers import get_trunk_output_feature_names
from vissl.trainer.train_steps import get_train_step
from vissl.utils.distributed_utils import all_gather_heterogeneous, all_gather_sizes
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.io import save_file


def build_task(config):
    """Builds a ClassyTask from a config.

    This assumes a 'name' key in the config which is used to determine what
    task class to instantiate. For instance, a config `{"name": "my_task",
    "foo": "bar"}` will find a class that was registered as "my_task"
    (see :func:`register_task`) and call .from_config on it."""

    task = TASK_REGISTRY[config.TRAINER.TASK_NAME].from_config(config)
    return task


class SelfSupervisionTrainer(object):
    """
    The main entry point for any training or feature extraction workflows in VISSL.

    The trainer constructs a train_task which prepares all the components of the
    training (optimizer, loss, meters, model etc) using the settings specified by user
    in the yaml config file. See the vissl/trainer/train_task.py for more details.

    Args:
        cfg (AttrDict): user specified input config that has optimizer, loss, meters etc
                        settings relevant to the training
        dist_run_id (str): For multi-gpu training with PyTorch, we have to specify
                           how the gpus are going to rendezvous. This requires specifying
                           the communication method: file, tcp and the unique rendezvous
                           run_id that is specific to 1 run.
                           We recommend:
                                1) for 1node: use init_method=tcp and run_id=auto
                                2) for multi-node, use init_method=tcp and specify
                                run_id={master_node}:{port}
        checkpoint_path (str): if the training is being resumed from a checkpoint, path to
                          the checkpoint. The tools/run_distributed_engines.py automatically
                          looks for the checkpoint in the checkpoint directory.
        checkpoint_folder (str): what directory to use for checkpointing. The
                          tools/run_distributed_engines.py creates the directory based on user
                          input in the yaml config file.
        hooks (List[ClassyHooks]): the list of hooks to use during the training. The hooks
                          vissl/engines/{train, extract_features}.py determine the hooks.
    """

    def __init__(
        self,
        cfg: AttrDict,
        dist_run_id: str,
        checkpoint_path: str = None,
        checkpoint_folder: str = None,
        hooks: List[ClassyHook] = None,
    ):
        self.cfg = cfg
        self.dist_run_id = dist_run_id
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        self.setup_distributed(self.cfg.MACHINE.DEVICE == "gpu")

        # now we should build the task. The task will also have the State attached
        # to it. It will have information about phases (train, test) both. It will
        # also contain all the other information like optimizers, etc
        self.task = build_task(self.cfg)
        self.task.set_checkpoint_path(checkpoint_path)
        self.task.set_checkpoint_folder(checkpoint_folder)
        if hooks is None:
            hooks = []
        self.task.set_hooks(hooks)

    def setup_distributed(self, use_gpu: bool):
        """
        Setup the distributed training. VISSL support both GPU and CPU only training.
        (1) Initialize the torch.distributed.init_process_group if the distributed is
            not already initialized. The init_method, backend are specified by user in the
            yaml config file. See vissl/defaults.yaml file for description on how to set
            init_method, backend.
        (2) We also set the global cuda device index using torch.cuda.set_device or
            cpu device
        """
        # we overwrite the distributed trainer setup here with our config options
        distributed_world_size = int(os.environ["WORLD_SIZE"])
        assert distributed_world_size % self.cfg.DISTRIBUTED.NUM_NODES == 0
        init_method = f"{self.cfg.DISTRIBUTED.INIT_METHOD}://{self.dist_run_id}"
        logging.info(
            f"Using Distributed init method: {init_method}, "
            f"world_size: {distributed_world_size}, rank: {self.distributed_rank}"
        )

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.cfg.DISTRIBUTED.BACKEND,
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
            # perform a dummy all-reduce to initialize the NCCL communicator
            if torch.cuda.is_available() and (self.cfg.DISTRIBUTED.BACKEND == "nccl"):
                dist.all_reduce(torch.zeros(1).cuda())
        else:
            set_cpu_device()

    def train(self):
        """
        The train workflow. We get the training loop to use (vissl default is
        standard_train_step) but the user can create their own training loop
        and specify the name TRAINER.TRAIN_STEP_NAME

        The training happens:
        1. Execute any hooks at the start of training (mostly resets the variable like
           iteration num phase_num etc)
        2. For each epoch (train or test), run the hooks at the start of an epoch. Mostly
           involves setting things like timer, setting dataloader epoch etc
        3. Execute the training loop (1 training iteration) involving forward, loss, backward,
           optimizer update, metrics collection etc.
        4. At the end of epoch, sync meters and execute hooks at the end of phase. Involves
           things like checkpointing model, logging timers, logging to tensorboard etc
        """
        train_step_fn = get_train_step(self.cfg["TRAINER"]["TRAIN_STEP_NAME"])
        self.task.prepare(pin_memory=self.cfg.DATA.PIN_MEMORY)
        self.task.init_distributed_data_parallel_model()

        # Find what phase, train_phase_idx, local_iteration_num we are starting from.
        # Recover it from the checkpoint (if available)
        task, phase_idx, iteration_num = self._init_training_state(self.cfg, self.task)

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
                    if self.cfg.MODEL.CUDA_CACHE.CLEAR_CUDA_CACHE and (
                        iteration_num % self.cfg.MODEL.CUDA_CACHE.CLEAR_FREQ == 0
                    ):
                        logging.info(
                            f"Emptying CUDA cache at step count: {iteration_num}"
                        )
                        torch.cuda.empty_cache()
                        logging.info("CUDA cache cleared")
                    task = train_step_fn(task)
                    iteration_num += 1
                    task.local_iteration_num = iteration_num
                    # Book-keeping: update the training iteration number (only updated
                    # if it's a training phase).
                    task.iteration += 1 if task.train else 0
                    # Book-keeping. Track how many forward passes have been done.
                    # aka how many batches have been seen by the trainer irrespective of
                    # the train or test phase.
                    task.batches += 1
                    # update the batch time aka the training time for the current iteration.
                    task.batch_time.append(time.time() - task.start_time)
                    task.start_time = time.time()
                    task.run_hooks(SSLClassyHookFunctions.on_step.name)
                except StopIteration:
                    break
                except Exception as e:
                    task.run_hooks(SSLClassyHookFunctions.on_exception.name)
                    raise e
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
    def _init_training_state(cfg, task: ClassyTask) -> Tuple[ClassyTask, int, int]:
        """
        If a checkpoint is present, recover the current training status.
        If not initialize everything properly

        Args:
            task {ClassyTask}: object consisting of all components a training requires
                               (meters, optimizers, model, loss etc.)

        Returns:
            task {ClassyTask}: updated task
            phase_idx {int}: phase index
            iteration_num: iteration number
        """

        phase_idx, iteration_num = -1, -1

        # Ensure that train loader exists. Will NOT exist if config.TEST_ONLY is True
        if "train" in task.dataloaders.keys():
            loader_key = "train"
        else:
            loader_key = "test"

        task.max_iteration = task.num_train_phases * len(task.dataloaders[loader_key])

        if task.checkpoint is not None:
            phase_idx = task.checkpoint["phase_idx"]
            task.train_phase_idx = task.checkpoint["train_phase_idx"]
            task.local_iteration_num = task.checkpoint["iteration_num"]
            task.iteration = task.checkpoint["iteration"]
        else:
            task.iteration = 0
            task.local_iteration_num = iteration_num

        num_iter_in_phase = len(task.dataloaders[loader_key])
        num_iter_in_epoch = num_iter_in_phase * task.num_train_phases_per_epoch

        num_samples = task.num_phase_samples(loader_key)
        task.start_time = time.time()
        task.batch_time = []
        task.metrics = {}
        logging.info(f"Training {task.num_epochs} epochs")
        logging.info(f"One epoch = {num_iter_in_epoch} iterations.")
        logging.info(f"Total {num_samples} samples in one epoch")

        if task.num_epochs != task.num_train_phases:
            logging.info(f"Training a total of {task.num_train_phases} train phases.")
            logging.info(f"One phase = {num_iter_in_phase} iterations.")

        logging.info(f"Total {task.max_iteration} iterations for training")

        return task, phase_idx, task.local_iteration_num

    def _advance_phase(self, task: ClassyTask):
        """
        Advance the training phase to the next phase.
        - Updates the phase number,
        - resets the meters,
        - reset losses,
        - recreates the data iterator and destroys previous iterator
        - set the model to be in train or eval phase depending on what phase we are in
        - execute any optimizer update (normally learning rate updates etc at the end of
          an epoch)
        """
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

        # get a new data iterator - delete the iterator at the beginning explicitly
        # so that all dataloader processes are cleaned up
        phase_type = "train" if phase["train"] else "test"
        # we are advancing to next epoch, so no need to compute start_iter,
        # just let it to be 0 inside of recreate_data_iterator. However, if we are just
        # starting from the resumed training, we want to compute_start_iter
        # again (if applicable) since we recreate the data iterator and delete
        # the old ones.
        compute_start_iter = False
        if task.checkpoint is not None and task.checkpoint["train_phase_idx"] == (
            task.train_phase_idx - 1
        ):
            compute_start_iter = True

        task.recreate_data_iterator(
            phase_type,
            epoch=task.phase_idx,
            compute_start_iter=compute_start_iter,
            train_phase_idx=task.train_phase_idx,
        )

        # set the model to train or eval depending on what phase we are in
        task.model.train(phase["train"])

        if task.train and task.train_phase_idx >= 0:
            task.optimizer.on_epoch(task.where)

        local_rank, _ = get_machine_local_and_dist_rank()
        logging.info(f"Phase advanced. Rank: {local_rank}")

    def extract(self, output_folder: str) -> None:
        """
        Extract workflow supports multi-gpu feature extraction. Since we are only extracting
        features, only the model is built (and initialized from some model weights file
        if specified by user). The model is set to the eval mode fully.

        The features are extracted for whatever data splits (train, val, test) etc that user
        wants.
        """
        # support feature extraction on gpu only.
        assert self.task.device.type == "cuda", "Set MACHINE.DEVICE = gpu"
        self.task.prepare_extraction(pin_memory=self.cfg.DATA.PIN_MEMORY)

        # Create distributed model
        self._add_dummy_layer()
        self.task.init_distributed_data_parallel_model()
        if is_primary():
            logging.info("Model is:\n {}".format(self.task.model))

        # Get the names of the features that we are extracting. If user doesn't
        # specify the features to evaluate, we get the full model output and freeze
        # head/trunk both as caution.
        feat_names = get_trunk_output_feature_names(self.cfg.MODEL)
        if len(feat_names) == 0:
            feat_names = ["heads"]

        for split in self.task.available_splits:
            logging.info(f"============== Split: {split} =======================")
            logging.info(f"Extracting features for partition: {split.lower()}")
            self.task.data_iterator = iter(self.task.dataloaders[split.lower()])
            self._extract_split_features(feat_names, self.task, split, output_folder)
            logging.info(f"Done getting features for partition: {split.lower()}")

        self._cleanup_task()

    @staticmethod
    def _flatten_features_list(features: Dict[str, Any]):
        assert isinstance(features, list), "features must be of type list"
        is_nested = isinstance(features[0], list)
        if is_nested:
            flat_features_list = [item for sublist in features for item in sublist]
            return flat_features_list
        return features

    @staticmethod
    def _save_extracted_features(
        features,
        targets,
        dist_rank: int,
        chunk_index: int,
        split: str,
        output_folder: str,
    ):
        output = {}
        for layer_name in features.keys():
            indices = sorted(features[layer_name].keys())
            if len(indices) > 0:
                output[layer_name] = {
                    "inds": np.array(indices),
                    "features": np.array([features[layer_name][i] for i in indices]),
                    "targets": np.array([targets[layer_name][i] for i in indices]),
                }

        for layer_name, layer_features in output.items():
            out_feat_file = os.path.join(
                output_folder,
                f"rank{dist_rank}_chunk{chunk_index}_{split.lower()}_{layer_name}_features.npy",
            )
            out_target_file = os.path.join(
                output_folder,
                f"rank{dist_rank}_chunk{chunk_index}_{split.lower()}_{layer_name}_targets.npy",
            )
            out_inds_file = os.path.join(
                output_folder,
                f"rank{dist_rank}_chunk{chunk_index}_{split.lower()}_{layer_name}_inds.npy",
            )
            save_file(layer_features["features"], out_feat_file)
            save_file(layer_features["targets"], out_target_file)
            save_file(layer_features["inds"], out_inds_file)

    def _extract_split_features(
        self,
        feat_names: List[str],
        task: ClassyTask,
        split_name: str,
        output_folder: str,
    ):
        task.model.eval()
        logging.info("Model set to eval mode during feature extraction...")
        dist_rank = torch.distributed.get_rank()

        out_features, out_targets = {}, {}
        for feat_name in feat_names:
            out_features[feat_name], out_targets[feat_name] = {}, {}

        chunk_index = 0
        feature_buffer_size = 0
        while True:
            try:
                sample = next(task.data_iterator)
                assert isinstance(sample, dict)
                assert "data_idx" in sample, "Indices not passed"
                input_sample = {
                    "input": torch.cat(sample["data"]).cuda(non_blocking=True),
                    "target": torch.cat(sample["label"]).cpu().numpy(),
                    "inds": torch.cat(sample["data_idx"]).cpu().numpy(),
                }
                with torch.no_grad():
                    features = task.model(input_sample["input"])
                    flat_features_list = self._flatten_features_list(features)
                    num_images = input_sample["inds"].shape[0]
                    feature_buffer_size += num_images
                    for num, feat_name in enumerate(feat_names):
                        feature = flat_features_list[num].cpu().numpy()
                        targets = input_sample["target"]
                        for idx in range(num_images):
                            index = input_sample["inds"][idx]
                            out_features[feat_name][index] = feature[idx]
                            out_targets[feat_name][index] = targets[idx].reshape(-1)

                if (
                    feature_buffer_size
                    >= self.cfg.EXTRACT_FEATURES.CHUNK_THRESHOLD
                    >= 0
                ):
                    self._save_extracted_features(
                        features=out_features,
                        targets=out_targets,
                        dist_rank=dist_rank,
                        chunk_index=chunk_index,
                        split=split_name,
                        output_folder=output_folder,
                    )
                    for layer_name in out_features.keys():
                        out_features[layer_name].clear()
                    chunk_index += 1
                    feature_buffer_size = 0

            except StopIteration:
                self._save_extracted_features(
                    features=out_features,
                    targets=out_targets,
                    dist_rank=dist_rank,
                    chunk_index=chunk_index,
                    split=split_name,
                    output_folder=output_folder,
                )
                break

    def _add_dummy_layer(self):
        """
        In case of feature evaluation mode, if we are freezing both trunk and
        head, DDP won't work as there are no parameters in the model. Adding
        the dummy head will lead to features being not right. So we rather
        add the dummy layer to the model and use DDP. We copy the model to
        gpu (if using gpus) after the new dummy layer addition.
        """
        fully_frozen_model = self.task.base_model.is_fully_frozen_model()
        if fully_frozen_model:
            self.task.base_model.dummy_layer = torch.nn.Linear(4, 4)
            if self.task.device.type == "cuda":
                self.task.base_model = copy_model_to_gpu(self.task.base_model)

    def _cleanup_task(self):
        if hasattr(self.task, "data_iterator"):
            del self.task.data_iterator
            gc.collect()
        if hasattr(self.task, "dataloaders"):
            del self.task.dataloaders
            gc.collect()

    def extract_clusters(self) -> Dict[str, Dict[int, int]]:
        """
        Workflow to extract multi-gpu cluster extraction for pre-trained models
        based on clusterization (SwAV, DeepCluster, etc).

        The function returns a map from image index to cluster index for the
        whole dataset for each of the different splits.
        """

        # Support feature extraction on gpu only.
        assert self.task.device.type == "cuda", "Set MACHINE.DEVICE = gpu"
        self.task.prepare_extraction(pin_memory=self.cfg.DATA.PIN_MEMORY)

        # Assert that the model support extract of clusters
        error_message = "Extracting clusters is only available for pre-training methods based on clusters"  # NOQA
        assert self.task.base_model.is_clustering_model(), error_message

        # Create distributed model
        self._add_dummy_layer()
        self.task.init_distributed_data_parallel_model()
        if is_primary():
            logging.info("Model is:\n {}".format(self.task.model))

        # Compute the cluster assignment on each worker in parallel
        cluster_assignment = {}
        for split in self.task.available_splits:
            msg = f"Extracting cluster assignment for partition: {split}"
            logging.info(msg)
            cluster_assignment[split] = self._get_cluster_assignment_for_split(
                self.task, split
            )
            logging.info("Done: " + msg)
        self._cleanup_task()

        # Merge the cluster assignments and group by cluster
        return self._merge_cluster_assignments(cluster_assignment)

    def _get_cluster_assignment_for_split(self, task: ClassyTask, split: str):
        task.model.eval()
        logging.info("Model set to eval mode during feature extraction...")

        cluster_assignments = {}
        task.data_iterator = iter(self.task.dataloaders[split.lower()])
        while True:
            try:
                sample = next(task.data_iterator)
                assert isinstance(sample, dict)
                assert "data_idx" in sample, "Indices not passed"

                input_sample = {
                    "images": torch.cat(sample["data"]).cuda(non_blocking=True),
                    "indices": torch.cat(sample["data_idx"]).cpu().numpy(),
                }

                with torch.no_grad():
                    features = task.model(input_sample["images"])
                    features = features[0]
                    prototype_score = features[1]
                    prototype_index = prototype_score.argmax(dim=-1)
                    num_images = input_sample["indices"].shape[0]
                    for idx in range(num_images):
                        image_index = input_sample["indices"][idx]
                        cluster_assignments[image_index] = prototype_index[idx].item()
            except StopIteration:
                break
        return cluster_assignments

    @staticmethod
    def _merge_cluster_assignments(
        rank_cluster_assignment: Dict[str, Dict[int, int]]
    ) -> Dict[str, Dict[int, int]]:
        """
        All gather all the cluster assignments computed by the different workers on
        separate parts of the dataset and merge them in a single map
        """
        merged_cluster_assignments = {}
        for split in rank_cluster_assignment.keys():

            split_assignments = list(rank_cluster_assignment[split].items())
            image_indices = [assignment[0] for assignment in split_assignments]
            image_indices = torch.LongTensor(image_indices).cuda(
                torch.cuda.current_device()
            )
            cluster_indices = [assignment[1] for assignment in split_assignments]
            cluster_indices = torch.LongTensor(cluster_indices).cuda(
                torch.cuda.current_device()
            )

            sizes = all_gather_sizes(image_indices)
            all_image_indices = all_gather_heterogeneous(sizes, image_indices)
            all_cluster_indices = all_gather_heterogeneous(sizes, cluster_indices)

            merged_cluster_assignments[split] = {}
            for image_indices, cluster_indices in zip(
                all_image_indices, all_cluster_indices
            ):
                for image_id, cluster_id in zip(image_indices, cluster_indices):
                    merged_cluster_assignments[split][
                        image_id.item()
                    ] = cluster_id.item()
        return merged_cluster_assignments
