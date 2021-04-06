# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script contains some helpful functions to handle wandb setup.
"""

import os
import logging

from classy_vision.generic.distributed_util import is_primary
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.io import makedir
from omegaconf import DictConfig

def is_wandb_available():
    """
    Check whether wandb is available or not.

    Returns:
        wandb_available (bool): based on wandb imports, returns whether tensboarboard
                             is available or not.
    """
    try:
        import wandb  # noqa F401

        wandb_available = True
    except ImportError:
        logging.info("Tensorboard is not available")
        wandb_available = False
    return wandb_available


def get_wandb_dir(cfg):
    """
    Get the output directory where the wandb events will be written.

    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        wandb as well like log directory, logging frequency etc

    Returns:
        wandb_dir (str): output directory path

    """
    checkpoint_folder = get_checkpoint_folder(cfg)
    wandb_dir = f"{checkpoint_folder}/wandb_logs"
    logging.info(f"Wandb Local dir: {wandb_dir}")
    makedir(wandb_dir)
    return wandb_dir


def parse_hydra_config(cfg):
    final = {}

    def recurse(key_array):
        current = cfg
        for key in key_array:
            current = getattr(current, key)

        if type(current) != DictConfig:
            final['.'.join(key_array)] = current
        else:
            for key in current:
                recurse(key_array + [key])

    recurse([])
    return final


def get_wandb_hook(cfg):
    """
    Construct the Tensorboard hook for visualization from the specified config

    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        wandb as well like log directory, logging frequency etc

    Returns:
        SSLTensorboardHook (function): the wandb hook constructed
    """
    import wandb
    from vissl.hooks import SSLWandbHook

    if is_primary():
        # it's important to only do this once in WandB, multiple init calls lead to
        # unexpected results

        # get the wandb directory and check wandb is installed
        wandb_dir = get_wandb_dir(cfg)

        # wandb relies on an environment variable to figure out where to dump
        # log files. we will overwrite this to make sure they are store within
        # the checkpoint directory
        os.environ['WANDB_DIR'] = wandb_dir

        # unlike Tensorboard, we need to handle preemption differently
        # best way to do that is to create a unique ID inside the directory file
        # and initialize the wandb constructor with it

        # if this run has already been initialized, we will fetch the wandb id
        wandb_id_save_path = os.path.join(wandb_dir, 'wandb_id.txt')

        if os.path.exists(wandb_id_save_path):
            wandb_id = open(wandb_id_save_path, 'r').read().splitlines()[0]
        else:
            wandb_id = wandb.util.generate_id()
            with open(wandb_id_save_path, 'w') as f:
                f.write(wandb_id)

        name = cfg.HOOKS.WANDB_SETUP.EXP_NAME
        if name == "??":
            logging.info('wandb exp name was not overwritten. using wandb random name')
            name = None

        wandb.init(id=wandb_id,
                   config=parse_hydra_config(cfg),
                   resume='allow',
                   project=cfg.HOOKS.WANDB_SETUP.PROJECT_NAME,
                   name=name
        )

    return SSLWandbHook(
        log_params=cfg.HOOKS.WANDB_SETUP.LOG_PARAMS,
        log_params_every_n_iterations=cfg.HOOKS.WANDB_SETUP.LOG_PARAMS_EVERY_N_ITERS,
        log_params_gradients=cfg.HOOKS.WANDB_SETUP.LOG_PARAMS_GRADIENTS,
    )


