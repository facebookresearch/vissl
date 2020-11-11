# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script contains some helpful functions to handle tensorboard setup.
"""

import logging

from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.io import makedir


def is_tensorboard_available():
    try:
        import tensorboard  # noqa F401
        from torch.utils.tensorboard import SummaryWriter  # noqa F401

        tb_available = True
    except ImportError:
        logging.info("Tensorboard is not available")
        tb_available = False
    return tb_available


def get_tensorboard_dir(cfg):
    checkpoint_folder = get_checkpoint_folder(cfg)
    tensorboard_dir = f"{checkpoint_folder}/tb_logs"
    logging.info(f"Tensorboard dir: {tensorboard_dir}")
    makedir(tensorboard_dir)
    return tensorboard_dir


def get_tensorboard_hook(cfg):
    from torch.utils.tensorboard import SummaryWriter
    from vissl.hooks import SSLTensorboardHook

    # get the tensorboard directory and check tensorboard is installed
    tensorboard_dir = get_tensorboard_dir(cfg)
    flush_secs = cfg.TENSORBOARD_SETUP.FLUSH_EVERY_N_MIN * 60
    log_activations = cfg.TENSORBOARD_SETUP.LOG_ACTIVATIONS
    return SSLTensorboardHook(
        tb_writer=SummaryWriter(log_dir=tensorboard_dir, flush_secs=flush_secs),
        log_activations=log_activations,
    )
