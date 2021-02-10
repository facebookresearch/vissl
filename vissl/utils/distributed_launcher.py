# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features
"""

import logging
import tempfile
from typing import Any, Callable, List

import torch
from fvcore.common.file_io import PathManager
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
from vissl.utils.hydra_config import AttrDict
from vissl.utils.io import cleanup_dir, copy_data_to_local, makedir
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import get_dist_run_id
from vissl.utils.slurm import get_node_id


def _get_available_splits(cfg: AttrDict):
    return [key for key in cfg.DATA if key.lower() in ["train", "test"]]


def _copy_to_local(cfg: AttrDict):
    available_splits = _get_available_splits(cfg)
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


def _cleanup_local_dir(cfg: AttrDict):
    available_splits = _get_available_splits(cfg)
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
    Launch the distributed training across gpus of the current node according to the cfg.

    If more than 1 nodes are needed for training, this function should be called on each
    of the different nodes, each time with an unique node_id in the range [0..N-1] if N
    is the total number of nodes to take part in training.

    Alternatively, you can use SLURM or any cluster management system to run this function
    for you.

    Configure the node_id, dist_run_id, setup the environment variabled

    Args:
        cfg (AttrDict): VISSL yaml configuration
        node_id (int): node_id for this node
        engine_name (str): what engine to run: train or extract_features
        hook_generator (Callable): Callback to generate all the ClassyVision hooks
            for this engine
    """

    setup_logging(__name__)
    node_id = get_node_id(node_id)
    dist_run_id = get_dist_run_id(cfg, cfg.DISTRIBUTED.NUM_NODES)
    world_size = cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    set_env_vars(local_rank=0, node_id=node_id, cfg=cfg)
    _copy_to_local(cfg)

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
        _cleanup_local_dir(cfg)

    logging.info("All Done!")
    shutdown_logging()


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


class _ResumableSlurmJob:
    def __init__(self, engine_name: str, config: AttrDict):
        self.engine_name = engine_name
        self.config = config

    def __call__(self):
        import submitit

        environment = submitit.JobEnvironment()
        node_id = environment.global_rank
        master_ip = environment.hostnames[0]
        master_port = self.config.SLURM.PORT_ID
        self.config.DISTRIBUTED.INIT_METHOD = "tcp"
        self.config.DISTRIBUTED.RUN_ID = f"{master_ip}:{master_port}"
        launch_distributed(
            cfg=self.config,
            node_id=node_id,
            engine_name=self.engine_name,
            hook_generator=default_hook_generator,
        )

    def checkpoint(self):
        import submitit

        trainer = _ResumableSlurmJob(engine_name=self.engine_name, config=self.config)
        return submitit.helpers.DelayedSubmission(trainer)


def launch_distributed_on_slurm(cfg: AttrDict, engine_name: str):
    """
    Launch a distributed training on SLURM, allocating the nodes and GPUs as described in
    the configuration, and calls the function "launch_on_local_node" appropriately on each
    of the nodes.

    Args:
        cfg (AttrDict): the configuration of the experiment
        engine_name (str): the name of the engine to run (train or extract_features)
    """
    ######################################################################################
    # DO NOT REMOVE: submitit processes will not be initialized correctly if numpy is not
    # imported first
    ######################################################################################
    import numpy

    print(numpy.__version__)

    import submitit

    # setup the log folder
    log_folder = cfg.SLURM.LOG_FOLDER
    makedir(log_folder)
    assert PathManager.exists(
        log_folder
    ), f"Specified config.SLURM.LOG_FOLDER={log_folder} doesn't exist"

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        name=cfg.SLURM.NAME,
        slurm_comment=cfg.SLURM.COMMENT,
        slurm_partition=cfg.SLURM.PARTITION,
        slurm_constraint=cfg.SLURM.CONSTRAINT,
        timeout_min=cfg.SLURM.TIME_HOURS * 60,
        nodes=cfg.DISTRIBUTED.NUM_NODES,
        cpus_per_task=8 * cfg.DISTRIBUTED.NUM_PROC_PER_NODE,
        tasks_per_node=1,
        gpus_per_node=cfg.DISTRIBUTED.NUM_PROC_PER_NODE,
        mem_gb=cfg.SLURM.MEM_GB,
    )
    trainer = _ResumableSlurmJob(engine_name=engine_name, config=cfg)
    job = executor.submit(trainer)
    print(f"SUBMITTED: {job.job_id}")
