#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os

import numpy as np
from torch.utils.collect_env import get_pretty_env_info
from vissl.ssl_tasks import build_task
from vissl.ssl_trainer import DistributedSelfSupervisionTrainer
from vissl.utils.checkpoint import get_absolute_path
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import print_cfg
from vissl.utils.logger import setup_logging
from vissl.utils.misc import set_seeds, setup_multiprocessing_method


def extract_main(args, cfg, dist_run_id, local_rank=0, node_id=0):
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
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print_cfg(cfg)
        logging.info("System config:\n{}".format(get_pretty_env_info()))

    ssl_task = build_task(cfg)
    trainer = DistributedSelfSupervisionTrainer(dist_run_id)
    output_dir = get_absolute_path(cfg.SVM.OUTPUT_DIR)

    features = trainer.extract(cfg, ssl_task)
    for split in features.keys():
        logging.info(f"============== Split: {split} =======================")
        layers = features[split].keys()
        for layer in layers:
            out_feat_file = os.path.join(
                output_dir, f"rank{local_rank}_{split}_{layer}_features.npy"
            )
            out_target_file = os.path.join(
                output_dir, f"rank{local_rank}_{split}_{layer}_targets.npy"
            )
            out_inds_file = os.path.join(
                output_dir, f"rank{local_rank}_{split}_{layer}_inds.npy"
            )
            logging.info(
                "Saving extracted features: {} {} to: {}".format(
                    layer, features[split][layer]["features"].shape, out_feat_file
                )
            )
            np.save(out_feat_file, features[split][layer]["features"])
            logging.info(
                "Saving extracted targets: {} to: {}".format(
                    features[split][layer]["targets"].shape, out_target_file
                )
            )
            np.save(out_target_file, features[split][layer]["targets"])
            logging.info(
                "Saving extracted indices: {} to: {}".format(
                    features[split][layer]["inds"].shape, out_inds_file
                )
            )
            np.save(out_inds_file, features[split][layer]["inds"])

    logging.info("All Done!")
