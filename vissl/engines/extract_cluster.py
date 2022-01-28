# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Callable, List

import torch
from classy_vision.hooks import ClassyHook
from vissl.config import AttrDict
from vissl.engines.engine_registry import Engine, register_engine
from vissl.hooks import default_hook_generator
from vissl.trainer import SelfSupervisionTrainer
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.cluster_utils import ClusterAssignment, ClusterAssignmentLoader
from vissl.utils.collect_env import collect_env_info
from vissl.utils.env import (
    get_machine_local_and_dist_rank,
    print_system_env_info,
    set_env_vars,
)
from vissl.utils.hydra_config import print_cfg
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import set_seeds, setup_multiprocessing_method


@register_engine("extract_cluster")
class ExtractClusterEngine(Engine):
    def run_engine(
        self,
        cfg: AttrDict,
        dist_run_id: str,
        checkpoint_path: str,
        checkpoint_folder: str,
        local_rank: int = 0,
        node_id: int = 0,
        hook_generator: Callable[[Any], List[ClassyHook]] = default_hook_generator,
    ):
        extract_clusters(
            cfg, dist_run_id, checkpoint_folder, local_rank=local_rank, node_id=node_id
        )


def extract_clusters(
    cfg: AttrDict,
    dist_run_id: str,
    checkpoint_folder: str,
    local_rank: int = 0,
    node_id: int = 0,
):
    """
    Sets up and executes model visualisation extraction workflow on one node
    """

    # setup the environment variables
    set_env_vars(local_rank, node_id, cfg)
    dist_rank = int(os.environ["RANK"])

    # setup logging
    setup_logging(__name__, output_dir=checkpoint_folder, rank=dist_rank)

    logging.info(f"Env set for rank: {local_rank}, dist_rank: {dist_rank}")
    # print the environment info for the current node
    if local_rank == 0:
        current_env = os.environ.copy()
        print_system_env_info(current_env)

    # setup the multiprocessing to be forkserver.
    # See https://fb.quip.com/CphdAGUaM5Wf
    setup_multiprocessing_method(cfg.MULTI_PROCESSING_METHOD)

    # set seeds
    logging.info("Setting seed....")
    set_seeds(cfg, dist_rank)

    # We set the CUDA device here as well as a safe solution for all downstream
    # `torch.cuda.current_device()` calls to return correct device.
    if cfg.MACHINE.DEVICE == "gpu" and torch.cuda.is_available():
        local_rank, _ = get_machine_local_and_dist_rank()
        torch.cuda.set_device(local_rank)

    # print the training settings and system settings
    if local_rank == 0:
        print_cfg(cfg)
        logging.info("System config:\n{}".format(collect_env_info()))

    # Build the SSL trainer to set up distributed training and then
    # extract the cluster assignments for all entries in the dataset
    trainer = SelfSupervisionTrainer(cfg, dist_run_id)
    output_folder = get_checkpoint_folder(cfg)
    cluster_assignments = trainer.extract_clusters(output_folder=output_folder)

    # Save the cluster assignments in the output folder
    if dist_rank == 0:
        assignment = ClusterAssignment(
            config=cfg, cluster_assignments=cluster_assignments
        )
        ClusterAssignmentLoader.save_cluster_assignment(
            output_dir=output_folder, assignments=assignment
        )
        ClusterAssignmentLoader.save_cluster_assignment_as_dataset(
            output_dir=output_folder, assignments=assignment
        )

    # close the logging streams including the file handlers
    logging.info("All Done!")
    shutdown_logging()
