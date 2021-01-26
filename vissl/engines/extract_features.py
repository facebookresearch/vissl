# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

from vissl.trainer import SelfSupervisionTrainer
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.collect_env import collect_env_info
from vissl.utils.env import get_machine_local_and_dist_rank, set_env_vars
from vissl.utils.hydra_config import AttrDict, print_cfg
from vissl.utils.io import save_file
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import set_seeds, setup_multiprocessing_method


def extract_main(
    cfg: AttrDict, dist_run_id: str, local_rank: int = 0, node_id: int = 0
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

    # setup logging
    setup_logging(__name__)
    # setup the environment variables
    set_env_vars(local_rank, node_id, cfg)

    # setup the multiprocessing to be forkserver.
    # See https://fb.quip.com/CphdAGUaM5Wf
    setup_multiprocessing_method(cfg.MULTI_PROCESSING_METHOD)

    # set seeds
    logging.info("Setting seed....")
    set_seeds(cfg)

    # print the training settings and system settings
    local_rank, _ = get_machine_local_and_dist_rank()
    if local_rank == 0:
        print_cfg(cfg)
        logging.info("System config:\n{}".format(collect_env_info()))

    output_dir = get_checkpoint_folder(cfg)
    trainer = SelfSupervisionTrainer(cfg, dist_run_id)
    features = trainer.extract()

    for split in features.keys():
        logging.info(f"============== Split: {split} =======================")
        layers = features[split].keys()
        for layer in layers:
            out_feat_file = (
                f"{output_dir}/rank{local_rank}_{split}_{layer}_features.npy"
            )
            out_target_file = (
                f"{output_dir}/rank{local_rank}_{split}_{layer}_targets.npy"
            )
            out_inds_file = f"{output_dir}/rank{local_rank}_{split}_{layer}_inds.npy"
            logging.info(
                "Saving extracted features: {} {} to: {}".format(
                    layer, features[split][layer]["features"].shape, out_feat_file
                )
            )
            save_file(features[split][layer]["features"], out_feat_file)
            logging.info(
                "Saving extracted targets: {} to: {}".format(
                    features[split][layer]["targets"].shape, out_target_file
                )
            )
            save_file(features[split][layer]["targets"], out_target_file)
            logging.info(
                "Saving extracted indices: {} to: {}".format(
                    features[split][layer]["inds"].shape, out_inds_file
                )
            )
            save_file(features[split][layer]["inds"], out_inds_file)
    logging.info("All Done!")
    # close the logging streams including the filehandlers
    shutdown_logging()
