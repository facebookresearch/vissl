# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features
"""

import logging
from argparse import Namespace
from typing import Any, Callable, List

import hydra
import torch
from omegaconf import DictConfig
from vissl.data.dataset_catalog import get_data_files
from vissl.engine.extract_features import extract_main
from vissl.engine.train import train_main
from vissl.ssl_hooks import ClassyHook, default_hook_generator
from vissl.utils.hydra_config import AttrDict, convert_to_attrdict, is_hydra_available
from vissl.utils.io import cleanup_dir, copy_data_to_local
from vissl.utils.logger import setup_logging
from vissl.utils.misc import get_dist_run_id
from vissl.utils.slurm import get_node_id


def get_available_splits(cfg):
    return [key for key in cfg.DATA if key.lower() in ["train", "test"]]


def copy_to_local(cfg):
    available_splits = get_available_splits(cfg)
    for split in available_splits:
        if cfg.DATA[split].COPY_TO_LOCAL_DISK:
            dest_dir = cfg.DATA[split]["COPY_DESTINATION_DIR"]
            assert not (dest_dir == "None") and (
                len(dest_dir) > 0
            ), f"Unknown copy location: {dest_dir}"
            data_files, label_files = get_data_files(split, cfg.DATA)
            data_files.extend(label_files)
            copy_data_to_local(data_files, dest_dir)


def cleanup_local_dir(cfg):
    available_splits = get_available_splits(cfg)
    for split in available_splits:
        if cfg.DATA[split].COPY_TO_LOCAL_DISK:
            dest_dir = cfg.DATA[split]["COPY_DESTINATION_DIR"]
            cleanup_dir(dest_dir)


def launch_distributed(
    cfg: AttrDict, args: Namespace, hook_generator: Callable[[Any], List[ClassyHook]]
):
    """Launch the distributed training across nodes, according to the config

    Args:
        cfg  -- VISSL configuration
        args -- Extra arguments for this node
        hook_generator -- Callback to generate all the ClassyVision hooks for this training
    """
    copy_to_local(cfg)
    node_id = get_node_id(args)
    dist_run_id = get_dist_run_id(cfg, cfg.DISTRIBUTED.NUM_NODES)
    world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE

    try:
        if world_size > 1:
            torch.multiprocessing.spawn(
                _distributed_worker,
                nprocs=cfg.DISTRIBUTED.NUM_PROC_PER_NODE,
                args=(cfg, node_id, dist_run_id, args, hook_generator),
                daemon=False,
            )
        else:
            _distributed_worker(
                local_rank=0,
                cfg=cfg,
                node_id=node_id,
                dist_run_id=dist_run_id,
                args=args,
                hook_generator=hook_generator,
            )

    except (KeyboardInterrupt, RuntimeError) as e:
        logging.error("Wrapping up, caught exception: ", e)
        if isinstance(e, RuntimeError):
            raise e
    finally:
        cleanup_local_dir(cfg)

    logging.info("All Done!")


def _distributed_worker(
    local_rank,
    cfg,
    node_id: int,
    dist_run_id: int,
    args: Namespace,
    hook_generator: Callable[[Any], List[ClassyHook]],
):
    dist_rank = cfg.DISTRIBUTED.NUM_PROC_PER_NODE * node_id + local_rank
    if args.extract_features:
        process_main = extract_main
    else:

        def process_main(args, cfg, dist_run_id, local_rank, node_id):
            train_main(
                args,
                cfg,
                dist_run_id,
                local_rank=local_rank,
                node_id=node_id,
                hook_generator=hook_generator,
            )

    logging.info(
        f"Spawning process for node_id: {node_id}, local_rank: {local_rank}, "
        f"dist_rank: {dist_rank}, dist_run_id: {dist_run_id}"
    )
    process_main(args, cfg, dist_run_id, local_rank=local_rank, node_id=node_id)


@hydra.main(config_path="hydra_configs", config_name="defaults")
def hydra_main(cfg: DictConfig):
    setup_logging(__name__)
    args, config = convert_to_attrdict(cfg)
    launch_distributed(config, args, hook_generator=default_hook_generator)


if __name__ == "__main__":
    assert is_hydra_available(), "Make sure to install hydra"
    hydra_main()
