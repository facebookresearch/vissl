# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Callable, List

import torch
from classy_vision.hooks import ClassyHook
from vissl.config import AttrDict
from vissl.engines.engine_registry import Engine, register_engine
from vissl.hooks import default_hook_generator
from vissl.trainer import SelfSupervisionTrainer
from vissl.utils.collect_env import collect_env_info
from vissl.utils.env import (
    get_machine_local_and_dist_rank,
    print_system_env_info,
    set_env_vars,
)
from vissl.utils.hydra_config import print_cfg
from vissl.utils.io import save_file
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import set_seeds, setup_multiprocessing_method


@register_engine("extract_features")
class ExtractFeatureEngine(Engine):
    def run_engine(
        self,
        cfg: AttrDict,
        dist_run_id: str,
        checkpoint_path: str,
        checkpoint_folder: str,
        local_rank: int = 0,
        node_id: int = 0,
        hook_generator: Callable[[Any], List[ClassyHook]] = default_hook_generator,
    ):
        extract_main(
            cfg, dist_run_id, checkpoint_folder, local_rank=local_rank, node_id=node_id
        )


def extract_main(
    cfg: AttrDict,
    dist_run_id: str,
    checkpoint_folder: str,
    local_rank: int = 0,
    node_id: int = 0,
):
    """
    Sets up and executes feature extraction workflow per machine.

    Args:
        cfg (AttrDict): user specified input config that has optimizer, loss, meters etc
                        settings relevant to the training
        dist_run_id (str): For multi-gpu training with PyTorch, we have to specify
                           how the gpus are going to rendezvous. This requires specifying
                           the communication method: file, tcp and the unique rendezvous
                           run_id that is specific to 1 run.
                           We recommend:
                                1) for 1node: use init_method=tcp and run_id=auto
                                2) for multi-node, use init_method=tcp and specify
                                run_id={master_node}:{port}
        local_rank (int): id of the current device on the machine. If using gpus,
                        local_rank = gpu number on the current machine
        node_id (int): id of the current machine. starts from 0. valid for multi-gpu
    """

    # setup the environment variables
    set_env_vars(local_rank, node_id, cfg)
    dist_rank = int(os.environ["RANK"])

    # setup logging
    setup_logging(__name__, output_dir=checkpoint_folder, rank=dist_rank)

    logging.info(f"Env set for rank: {local_rank}, dist_rank: {dist_rank}")
    # print the environment info for the current node
    if local_rank == 0:
        current_env = os.environ.copy()
        print_system_env_info(current_env)

    # setup the multiprocessing to be forkserver.
    # See https://fb.quip.com/CphdAGUaM5Wf
    setup_multiprocessing_method(cfg.MULTI_PROCESSING_METHOD)

    # set seeds
    logging.info("Setting seed....")
    set_seeds(cfg, dist_rank)

    # We set the CUDA device here as well as a safe solution for all downstream
    # `torch.cuda.current_device()` calls to return correct device.
    if cfg.MACHINE.DEVICE == "gpu" and torch.cuda.is_available():
        local_rank, _ = get_machine_local_and_dist_rank()
        torch.cuda.set_device(local_rank)

    # print the training settings and system settings
    if local_rank == 0:
        print_cfg(cfg)
        logging.info("System config:\n{}".format(collect_env_info()))

    trainer = SelfSupervisionTrainer(cfg, dist_run_id)
    features = trainer.extract()

    for split in features.keys():
        logging.info(f"============== Split: {split} =======================")
        for layer_name, layer_features in features[split].items():
            out_feat_file = os.path.join(
                checkpoint_folder, f"rank{dist_rank}_{split}_{layer_name}_features.npy"
            )
            out_target_file = os.path.join(
                checkpoint_folder, f"rank{dist_rank}_{split}_{layer_name}_targets.npy"
            )
            out_inds_file = os.path.join(
                checkpoint_folder, f"rank{dist_rank}_{split}_{layer_name}_inds.npy"
            )
            feat_shape = layer_features["features"].shape
            logging.info(
                f"Saving extracted features of {layer_name} with shape {feat_shape} to: {out_feat_file}"
            )
            save_file(layer_features["features"], out_feat_file)
            logging.info(
                f"Saving extracted targets of {layer_name} to: {out_target_file}"
            )
            save_file(layer_features["targets"], out_target_file)
            logging.info(
                f"Saving extracted indices of {layer_name} to: {out_inds_file}"
            )
            save_file(layer_features["inds"], out_inds_file)

    logging.info("All Done!")
    # close the logging streams including the filehandlers
    shutdown_logging()
