# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Callable, List

import torch
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.config import AttrDict
from vissl.engines.engine_registry import Engine, register_engine
from vissl.hooks import default_hook_generator
from vissl.trainer import SelfSupervisionTrainer
from vissl.utils.collect_env import collect_env_info
from vissl.utils.env import (
    get_machine_local_and_dist_rank,
    print_system_env_info,
    set_env_vars,
)
from vissl.utils.hydra_config import print_cfg
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import set_seeds, setup_multiprocessing_method


@register_engine("train")
class TrainerEngine(Engine):
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
        train_main(
            cfg,
            dist_run_id,
            checkpoint_path,
            checkpoint_folder,
            local_rank=local_rank,
            node_id=node_id,
            hook_generator=hook_generator,
        )


def train_main(
    cfg: AttrDict,
    dist_run_id: str,
    checkpoint_path: str,
    checkpoint_folder: str,
    local_rank: int = 0,
    node_id: int = 0,
    hook_generator: Callable[[Any], List[ClassyHook]] = default_hook_generator,
):
    """
    Sets up and executes training workflow per machine.

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
        local_rank (int): id of the current device on the machine. If using gpus,
                        local_rank = gpu number on the current machine
        node_id (int): id of the current machine. starts from 0. valid for multi-gpu
        hook_generator (Callable): The utility function that prepares all the hoooks that will
                         be used in training based on user selection. Some basic hooks are used
                         by default.
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

    # get the hooks - these hooks are executed per replica
    hooks = hook_generator(cfg)

    # build the SSL trainer. The trainer first prepares a "task" object which
    # acts as a container for various things needed in a training: datasets,
    # dataloader, optimizers, losses, hooks, etc. "Task" will also have information
    # about phases (train, test) both. The trainer then sets up distributed
    # training.
    trainer = SelfSupervisionTrainer(
        cfg, dist_run_id, checkpoint_path, checkpoint_folder, hooks
    )
    trainer.train()
    logging.info("All Done!")
    # close the logging streams including the filehandlers
    shutdown_logging()
