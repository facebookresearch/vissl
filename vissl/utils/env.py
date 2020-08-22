# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os

from vissl.utils.hydra_config import AttrDict


def set_env_vars(local_rank: int, node_id: int, cfg: AttrDict):
    os.environ["WORLD_SIZE"] = str(
        cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    )
    dist_rank = cfg.DISTRIBUTED.NUM_PROC_PER_NODE * node_id + local_rank
    os.environ["RANK"] = str(dist_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    if cfg.DISTRIBUTED.NCCL_DEBUG:
        os.environ["NCCL_DEBUG"] = "INFO"


def print_system_env_info(current_env):
    keys = list(current_env.keys())
    keys.sort()
    for key in keys:
        logging.info("{}:\t{}".format(key, current_env[key]))


def get_machine_local_and_dist_rank():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed_rank = int(os.environ.get("RANK", 0))
    return local_rank, distributed_rank
