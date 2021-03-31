# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script contains some helpful functions to handle tensorboard setup.
"""

import logging
import os

from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.io import makedir


def is_tensorboard_available():
    """
    Check whether tensorboard is available or not.

    Returns:
        tb_available (bool): based on tensorboard imports, returns whether tensboarboard
                             is available or not.
    """
    try:
        import tensorboard  # noqa F401
        from torch.utils.tensorboard import SummaryWriter  # noqa F401

        tb_available = True
    except ImportError:
        logging.info("Tensorboard is not available")
        tb_available = False
    return tb_available


def get_tensorboard_dir(config):
    """
    Get the output directory where the tensorboard events will be written.

    Args:
        config (AttrDict): User specified config file containing the settings for the
                        tensorboard as well like log directory, logging frequency etc

    Returns:
        tensorboard_dir (str): output directory path

    """
    tensorboard_dir = os.path.join(
        config.HOOKS.TENSORBOARD_SETUP.LOG_DIR,
        config.HOOKS.TENSORBOARD_SETUP.EXPERIMENT_LOG_DIR
    )
    if config.DISTRIBUTED.NUM_NODES > 1 and config.CHECKPOINT.APPEND_DISTR_RUN_ID:
        tensorboard_dir = f"{tensorboard_dir}/{config.DISTRIBUTED.RUN_ID}"
    logging.info(f"Tensorboard dir: {tensorboard_dir}")
    makedir(tensorboard_dir)
    return tensorboard_dir


def get_tensorboard_hook(cfg):
    """
    Construct the Tensorboard hook for visualization from the specified config

    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        tensorboard as well like log directory, logging frequency etc

    Returns:
        SSLTensorboardHook (function): the tensorboard hook constructed
    """

    # Instantiate the tensor board hook on the primary worker only
    #
    # Note: these checks are performed before torch.distributed is
    # initialized in the trainer (ex: SelfSupervisionTrainer) and
    # this is why they are not based on torch.distributed
    world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    if world_size > 1:
        local_rank, distributed_rank = get_machine_local_and_dist_rank()
        if local_rank != 0 or distributed_rank != 0:
            return None

    from torch.utils.tensorboard import SummaryWriter
    from vissl.hooks import SSLTensorboardHook

    # get the tensorboard directory and check tensorboard is installed
    tensorboard_dir = get_tensorboard_dir(cfg)
    flush_secs = cfg.HOOKS.TENSORBOARD_SETUP.FLUSH_EVERY_N_MIN * 60
    return SSLTensorboardHook(
        tb_writer=SummaryWriter(log_dir=tensorboard_dir, flush_secs=flush_secs),
        log_params=cfg.HOOKS.TENSORBOARD_SETUP.LOG_PARAMS,
        log_params_every_n_iterations=cfg.HOOKS.TENSORBOARD_SETUP.LOG_PARAMS_EVERY_N_ITERS,
        log_params_gradients=cfg.HOOKS.TENSORBOARD_SETUP.LOG_PARAMS_GRADIENTS,
    )
