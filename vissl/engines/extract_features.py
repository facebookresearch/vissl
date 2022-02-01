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
from vissl.hooks.profiling_hook import CudaSynchronizeHook
from vissl.models.model_helpers import get_trunk_output_feature_names
from vissl.trainer import SelfSupervisionTrainer
from vissl.utils.collect_env import collect_env_info
from vissl.utils.env import (
    get_machine_local_and_dist_rank,
    print_system_env_info,
    set_env_vars,
)
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.utils.hydra_config import print_cfg
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
        extract_features_main(
            cfg, dist_run_id, checkpoint_folder, local_rank=local_rank, node_id=node_id
        )


def extract_features_main(
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
        checkpoint_folder (str): what directory to use for checkpointing. This folder
                                 will be used to output the extracted features as well
                                 in case config.EXTRACT_FEATURES.OUTPUT_DIR is not set
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

    # Identify the hooks to run for the extract label engine
    # TODO - we need to plug this better with the engine registry
    #  - we either need to use the global hooks registry
    #  - or we need to create specific hook registry by engine
    hooks = extract_features_hook_generator(cfg)

    # Run the label prediction extraction
    trainer = SelfSupervisionTrainer(cfg, dist_run_id, hooks=hooks)
    output_dir = cfg.EXTRACT_FEATURES.OUTPUT_DIR or checkpoint_folder
    trainer.extract(
        output_folder=cfg.EXTRACT_FEATURES.OUTPUT_DIR or checkpoint_folder,
        extract_features=True,
        extract_predictions=False,
    )

    # TODO (prigoyal): merge this function with _extract_features
    if dist_rank == 0 and cfg.EXTRACT_FEATURES.MAP_FEATURES_TO_IMG_NAME:
        # Get the names of the features that we extracted features for. If user doesn't
        # specify the features to evaluate, we get the full model output and freeze
        # head/trunk both as caution.
        layers = get_trunk_output_feature_names(cfg.MODEL)
        if len(layers) == 0:
            layers = ["heads"]
        available_splits = [item.lower() for item in trainer.task.available_splits]
        for split in available_splits:
            image_paths = trainer.task.datasets[split].get_image_paths()[0]
            for layer in layers:
                ExtractedFeaturesLoader.map_features_to_img_filepath(
                    image_paths=image_paths,
                    input_dir=output_dir,
                    split=split,
                    layer=layer,
                )

    logging.info("All Done!")
    # close the logging streams including the filehandlers
    shutdown_logging()


def extract_features_hook_generator(cfg: AttrDict) -> List[ClassyHook]:
    hooks = []
    if cfg.MODEL.FSDP_CONFIG.FORCE_SYNC_CUDA:
        hooks.append(CudaSynchronizeHook())
    return hooks
