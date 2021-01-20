# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os

from fvcore.common.file_io import PathManager, HTTPURLHandler
from vissl.utils.hydra_config import AttrDict


def set_env_vars(local_rank: int, node_id: int, cfg: AttrDict):
    """
    Set some environment variables like total number of gpus used in training,
    distributed rank and local rank of the current gpu, whether to print the
    nccl debugging info and tuning nccl settings.
    """
    os.environ["WORLD_SIZE"] = str(
        cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    )
    dist_rank = cfg.DISTRIBUTED.NUM_PROC_PER_NODE * node_id + local_rank
    os.environ["RANK"] = str(dist_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    if cfg.DISTRIBUTED.NCCL_DEBUG:
        os.environ["NCCL_DEBUG"] = "INFO"
    if cfg.DISTRIBUTED.NCCL_SOCKET_NTHREADS:
        logging.info(
            f"local_rank: {local_rank}, "
            f"using NCCL_SOCKET_NTHREADS: {cfg.DISTRIBUTED.NCCL_SOCKET_NTHREADS}"
        )
        os.environ["NCCL_SOCKET_NTHREADS"] = str(cfg.DISTRIBUTED.NCCL_SOCKET_NTHREADS)
    # register http handler to support reading the urls
    PathManager.register_handler(HTTPURLHandler(), allow_override=True)


def print_system_env_info(current_env):
    """
    Print information about user system environment where VISSL is running.
    """
    keys = list(current_env.keys())
    keys.sort()
    for key in keys:
        logging.info("{}:\t{}".format(key, current_env[key]))


def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed_rank = int(os.environ.get("RANK", 0))
    return local_rank, distributed_rank
