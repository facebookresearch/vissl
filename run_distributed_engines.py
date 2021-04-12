# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features
"""

import logging
import sys
import tempfile
from typing import Any, Callable, List

import torch
from fvcore.common.file_io import PathManager
from hydra.experimental import compose, initialize_config_module
from vissl.data.dataset_catalog import get_data_files
from vissl.engines.extract_features import extract_main
from vissl.engines.train import train_main
from vissl.hooks import ClassyHook, default_hook_generator
from vissl.utils.checkpoint import (
    get_checkpoint_folder,
    get_resume_checkpoint,
    is_training_finished,
)
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import AttrDict, convert_to_attrdict, is_hydra_available
from vissl.utils.io import cleanup_dir, copy_data_to_local
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import get_dist_run_id
from vissl.utils.slurm import get_node_id


def get_available_splits(cfg: AttrDict):
    return [key for key in cfg.DATA if key.lower() in ["train", "test"]]


def copy_to_local(cfg: AttrDict):
    available_splits = get_available_splits(cfg)
    for split in available_splits:
        if cfg.DATA[split].COPY_TO_LOCAL_DISK:
            dest_dir = cfg.DATA[split]["COPY_DESTINATION_DIR"]
            tmp_dest_dir = tempfile.mkdtemp()
            data_files, label_files = get_data_files(split, cfg.DATA)
            data_files.extend(label_files)
            _, output_dir = copy_data_to_local(
                data_files, dest_dir, tmp_destination_dir=tmp_dest_dir
            )
            cfg.DATA[split]["COPY_DESTINATION_DIR"] = output_dir


def cleanup_local_dir(cfg: AttrDict):
    available_splits = get_available_splits(cfg)
    for split in available_splits:
        if cfg.DATA[split].COPY_TO_LOCAL_DISK:
            dest_dir = cfg.DATA[split]["COPY_DESTINATION_DIR"]
            cleanup_dir(dest_dir)


def launch_distributed(
    cfg: AttrDict,
    node_id: int,
    engine_name: str,
    hook_generator: Callable[[Any], List[ClassyHook]],
):
    """
    Launch the distributed training across gpus, according to the cfg

    Args:
        cfg  -- VISSL yaml configuration
        node_id -- node_id for this node
        engine_name -- what engine to run: train or extract_features
        hook_generator -- Callback to generate all the ClassyVision hooks for this engine
    """
    node_id = get_node_id(node_id)
    dist_run_id = get_dist_run_id(cfg, cfg.DISTRIBUTED.NUM_NODES)
    world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    set_env_vars(local_rank=0, node_id=node_id, cfg=cfg)
    copy_to_local(cfg)

    # given the checkpoint folder, we check that there's not already a final checkpoint
    checkpoint_folder = get_checkpoint_folder(cfg)
    if is_training_finished(cfg, checkpoint_folder=checkpoint_folder):
        logging.info(f"Training already succeeded on node: {node_id}, exiting.")
        return

    # Get the checkpoint where to load from. The load_checkpoints function will
    # automatically take care of detecting whether it's a resume or not.
    symlink_checkpoint_path = f"{checkpoint_folder}/checkpoint.torch"
    if cfg.CHECKPOINT.USE_SYMLINK_CHECKPOINT_FOR_RESUME and PathManager.exists(
        symlink_checkpoint_path
    ):
        checkpoint_path = f"{checkpoint_folder}/checkpoint.torch"
    else:
        checkpoint_path = get_resume_checkpoint(
            cfg, checkpoint_folder=checkpoint_folder
        )

    try:
        if world_size > 1:
            torch.multiprocessing.spawn(
                _distributed_worker,
                nprocs=cfg.DISTRIBUTED.NUM_PROC_PER_NODE,
                args=(
                    cfg,
                    node_id,
                    dist_run_id,
                    engine_name,
                    checkpoint_path,
                    checkpoint_folder,
                    hook_generator,
                ),
                daemon=False,
            )
        else:
            _distributed_worker(
                local_rank=0,
                cfg=cfg,
                node_id=node_id,
                dist_run_id=dist_run_id,
                engine_name=engine_name,
                checkpoint_path=checkpoint_path,
                checkpoint_folder=checkpoint_folder,
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
    local_rank: int,
    cfg: AttrDict,
    node_id: int,
    dist_run_id: str,
    engine_name: str,
    checkpoint_path: str,
    checkpoint_folder: str,
    hook_generator: Callable[[Any], List[ClassyHook]],
):
    dist_rank = cfg.DISTRIBUTED.NUM_PROC_PER_NODE * node_id + local_rank
    if engine_name == "extract_features":
        process_main = extract_main
    else:

        def process_main(cfg, dist_run_id, local_rank, node_id):
            train_main(
                cfg,
                dist_run_id,
                checkpoint_path,
                checkpoint_folder,
                local_rank=local_rank,
                node_id=node_id,
                hook_generator=hook_generator,
            )

    logging.info(
        f"Spawning process for node_id: {node_id}, local_rank: {local_rank}, "
        f"dist_rank: {dist_rank}, dist_run_id: {dist_run_id}"
    )
    process_main(cfg, dist_run_id, local_rank=local_rank, node_id=node_id)


def hydra_main(overrides: List[Any]):
    print(f"####### overrides: {overrides}")
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    setup_logging(__name__)
    args, config = convert_to_attrdict(cfg)
    launch_distributed(
        config,
        node_id=args.node_id,
        engine_name=args.engine_name,
        hook_generator=default_hook_generator,
    )
    # close the logging streams including the filehandlers
    shutdown_logging()


if __name__ == "__main__":
    """
    Example usage:

    `python tools/run_distributed_engines.py config=test/integration_test/quick_simclr`
    """
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    overrides.append("hydra.verbose=true")
    hydra_main(overrides=overrides)
