# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper to call torch.distributed.launch to run multi-gpu trainings.
Supports two engines: train and extract_features
"""

import logging
import tempfile
from typing import Any, Callable, List

import torch
from iopath.common.file_io import g_pathmgr
from vissl.config import AttrDict
from vissl.data.dataset_catalog import get_data_files
from vissl.engines import run_engine
from vissl.hooks import ClassyHook, default_hook_generator
from vissl.utils.checkpoint import (
    get_checkpoint_folder,
    get_resume_checkpoint,
    is_training_finished,
)
from vissl.utils.env import set_env_vars
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

    # If using gpus, we check that the user has specified <= gpus available on user system.
    if cfg.MACHINE.DEVICE == "gpu":
        assert cfg.DISTRIBUTED.NUM_PROC_PER_NODE <= torch.cuda.device_count(), (
            f"User system doesn't have requested {cfg.DISTRIBUTED.NUM_PROC_PER_NODE} gpus "
            f"available. Number of gpus found on user system={torch.cuda.device_count()}. "
            "Please set the DISTRIBUTED.NUM_PROC_PER_NODE properly."
        )

    # set the environment variables including local rank, node id etc.
    set_env_vars(local_rank=0, node_id=node_id, cfg=cfg)

    # given the checkpoint folder, we check that there's not already a final checkpoint
    # and that if there already exists a final checkpoint and user is not overriding
    # to ignore the final checkpoint
    checkpoint_folder = get_checkpoint_folder(cfg)
    if is_training_finished(cfg, checkpoint_folder=checkpoint_folder):
        logging.info(f"Training already succeeded on node: {node_id}, exiting.")
        return

    # Get the checkpoint where to resume from. The get_resume_checkpoint function will
    # automatically take care of detecting whether it's a resume or not.
    symlink_checkpoint_path = f"{checkpoint_folder}/checkpoint.torch"
    if cfg.CHECKPOINT.USE_SYMLINK_CHECKPOINT_FOR_RESUME and g_pathmgr.exists(
        symlink_checkpoint_path
    ):
        checkpoint_path = f"{checkpoint_folder}/checkpoint.torch"
    else:
        checkpoint_path = get_resume_checkpoint(
            cfg, checkpoint_folder=checkpoint_folder
        )

    # assert that if the user set the PARAMS_FILE, it must exist and be valid.
    # we only use the PARAMS_FILE init if the checkpoint doesn't exist for the
    # given training. This ensures that if the same training resumes, then it
    # resumes from the checkpoint and not the weight init
    if checkpoint_path is None and cfg["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]:
        params_file = cfg["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]
        error_message = f"Specified PARAMS_FILE does NOT exist: {params_file}"
        assert g_pathmgr.exists(params_file), error_message

    # copy the data to local if user wants. This can speed up dataloading.
    _copy_to_local(cfg)

    try:
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
    logging.info(
        f"Spawning process for node_id: {node_id}, local_rank: {local_rank}, "
        f"dist_rank: {dist_rank}, dist_run_id: {dist_run_id}"
    )

    if cfg.REPRODUCIBILITY.CUDDN_DETERMINISTIC:
        logging.info("torch.backends.cudnn.deterministic = True")
        torch.backends.cudnn.deterministic = True

    run_engine(
        engine_name,
        cfg,
        dist_run_id,
        checkpoint_path,
        checkpoint_folder,
        local_rank=local_rank,
        node_id=node_id,
        hook_generator=hook_generator,
    )


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


def create_submitit_executor(cfg: AttrDict):
    """
    Utility function to create a SLURM submitit executor, which
    is able to schedule arbitrary functions on a SLURM cluster

    The configuration of the executor is derived from the SLURM part
    of the VISSL configuration provided as parameter
    """
    import submitit

    log_folder = cfg.SLURM.LOG_FOLDER
    makedir(log_folder)
    assert g_pathmgr.exists(
        log_folder
    ), f"Specified config.SLURM.LOG_FOLDER={log_folder} doesn't exist"
    assert cfg.SLURM.PARTITION, "SLURM.PARTITION must be set when using SLURM"

    executor = submitit.AutoExecutor(folder=log_folder)
    timeout_min = cfg.SLURM.TIME_HOURS * 60 + cfg.SLURM.TIME_MINUTES
    executor.update_parameters(
        name=cfg.SLURM.NAME,
        slurm_comment=cfg.SLURM.COMMENT,
        slurm_partition=cfg.SLURM.PARTITION,
        slurm_constraint=cfg.SLURM.CONSTRAINT,
        timeout_min=timeout_min,
        nodes=cfg.DISTRIBUTED.NUM_NODES,
        cpus_per_task=cfg.SLURM.NUM_CPU_PER_PROC * cfg.DISTRIBUTED.NUM_PROC_PER_NODE,
        tasks_per_node=1,
        gpus_per_node=cfg.DISTRIBUTED.NUM_PROC_PER_NODE,
        mem_gb=cfg.SLURM.MEM_GB,
        slurm_additional_parameters=cfg.SLURM.ADDITIONAL_PARAMETERS,
    )
    return executor


def launch_distributed_on_slurm(cfg: AttrDict, engine_name: str):
    """
    Launch a distributed training on SLURM, allocating the nodes and GPUs as described in
    the configuration, and calls the function "launch_on_local_node" appropriately on each
    of the nodes.

    Args:
        cfg (AttrDict): the configuration of the experiment
        engine_name (str): the name of the engine to run (train or extract_features)
    """
    executor = create_submitit_executor(cfg)
    trainer = _ResumableSlurmJob(engine_name=engine_name, config=cfg)
    job = executor.submit(trainer)
    print(f"SUBMITTED: {job.job_id}")
    return job
