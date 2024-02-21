# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script contains some helpful functions to handle tensorboard setup.
"""

import logging

from vissl.utils.checkpoint import get_checkpoint_folder
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

        tb_available = True
    except ImportError:
        logging.info("Tensorboard is not available")
        tb_available = False
    return tb_available


def get_tensorboard_dir(cfg):
    """
    Get the output directory where the tensorboard events will be written.

    Args:
        cfg (AttrDict): User specified config file containing the settings for the
                        tensorboard as well like log directory, logging frequency etc

    Returns:
        tensorboard_dir (str): output directory path

    """
    checkpoint_folder = get_checkpoint_folder(cfg)
    tensorboard_dir = f"{checkpoint_folder}/tb_logs"
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
        log_activation_statistics=cfg.MONITORING.MONITOR_ACTIVATION_STATISTICS,
    )
