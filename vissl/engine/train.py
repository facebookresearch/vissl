# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
from typing import Any, Callable, List

import torch
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.ssl_hooks import default_hook_generator
from vissl.ssl_tasks import build_task
from vissl.ssl_trainer import DistributedSelfSupervisionTrainer
from vissl.utils.checkpoint import (
    get_checkpoint_folder,
    get_resume_checkpoint,
    is_training_finished,
)
from vissl.utils.collect_env import collect_env_info
from vissl.utils.env import (
    get_machine_local_and_dist_rank,
    print_system_env_info,
    set_env_vars,
)
from vissl.utils.hydra_config import print_cfg
from vissl.utils.logger import setup_logging
from vissl.utils.misc import set_seeds, setup_multiprocessing_method


def train_main(
    args,
    cfg,
    dist_run_id,
    local_rank=0,
    node_id=0,
    hook_generator: Callable[[Any], List[ClassyHook]] = default_hook_generator,
):
    # setup the environment variables
    set_env_vars(local_rank, node_id, cfg)
    dist_rank = int(os.environ["RANK"])

    # setup logging
    output_dir = get_checkpoint_folder(cfg)
    setup_logging(__name__, output_dir=output_dir, rank=dist_rank)

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
    set_seeds(cfg, args.node_id)

    # We set the CUDA device here as well as a safe solution for all downstream
    # `torch.cuda.current_device()` calls to return correct device.
    if cfg.MACHINE.DEVICE == "gpu" and torch.cuda.is_available():
        local_rank, _ = get_machine_local_and_dist_rank()
        torch.cuda.set_device(local_rank)

    # print the training settings and system settings
    if local_rank == 0:
        print_cfg(cfg)
        logging.info("System config:\n{}".format(collect_env_info()))

    # given the checkpoint folder, we check that there's not already a final checkpoint
    if is_training_finished(cfg, checkpoint_folder=output_dir):
        logging.info("Training already succeeded on this machine, bailing out")
        return

    # now we should build the task. The task will also have the State attached
    # to it. It will have information about phases (train, test) both. It will
    # also contain all the other information like optimizers, etc
    ssl_task = build_task(cfg)

    # Get the checkpoint where to load from. The load_checkpoints function will
    # automatically takes care of detecting whether it"s a resume or not
    checkpoint = get_resume_checkpoint(cfg, checkpoint_folder=output_dir)
    ssl_task.set_checkpoint(checkpoint)

    # get the hooks - these hooks are executed per replica
    hooks = hook_generator(cfg)
    ssl_task.set_hooks(hooks)

    trainer = DistributedSelfSupervisionTrainer(dist_run_id)
    trainer.train(cfg, ssl_task)
    logging.info("All Done!")
