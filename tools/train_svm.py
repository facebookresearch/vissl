# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import multiprocessing as mp
import sys
from argparse import Namespace
from typing import Any, List

import numpy as np
from hydra.experimental import compose, initialize_config_module
from run_distributed_engines import launch_distributed
from vissl.hooks import default_hook_generator
from vissl.models.model_helpers import get_trunk_output_feature_names
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import (
    AttrDict,
    convert_to_attrdict,
    is_hydra_available,
    print_cfg,
)
from vissl.utils.io import load_file
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import merge_features
from vissl.utils.svm_utils.svm_trainer import SVMTrainer


def train_svm(cfg: AttrDict, output_dir: str, layername: str):
    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=cfg)

    # train the svm
    logging.info(f"Training SVM for layer: {layername}")
    trainer = SVMTrainer(cfg["SVM"], layer=layername, output_dir=output_dir)
    train_data = merge_features(output_dir, "train", layername, cfg)
    train_features, train_targets = train_data["features"], train_data["targets"]
    trainer.train(train_features, train_targets)

    # test the svm
    test_data = merge_features(output_dir, "test", layername, cfg)
    test_features, test_targets = test_data["features"], test_data["targets"]
    trainer.test(test_features, test_targets)
    logging.info("All Done!")


def main(args: Namespace, config: AttrDict):
    # setup logging
    setup_logging(__name__)

    # print the coniguration used
    print_cfg(config)

    assert config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON, (
        "Feature eval mode is not ON. Can't run train_svm. "
        "Set config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True "
        "in your config or from command line."
    )
    # extract the features
    launch_distributed(
        config,
        args.node_id,
        engine_name="extract_features",
        hook_generator=default_hook_generator,
    )

    # Get the names of the features that we extracted features for. If user doesn't
    # specify the features to evaluate, we get the full model output and freeze
    # head/trunk both as caution.
    layers = get_trunk_output_feature_names(config.MODEL)
    if len(layers) == 0:
        layers = ["heads"]

    output_dir = get_checkpoint_folder(config)
    running_tasks = [
        mp.Process(target=train_svm, args=(config, output_dir, layer))
        for layer in layers
    ]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()

    # collect the mAP stats for all the layers and report
    output_mAP = []
    for layer in layers:
        try:
            ap_file = f"{output_dir}/{layer}/test_ap.npy"
            output_mAP.append(round(100.0 * np.mean(load_file(ap_file)), 3))
        except Exception:
            output_mAP.append(-1)
    logging.info(f"AP for various layers:\n {layers}: {output_mAP}")
    # close the logging streams including the filehandlers
    shutdown_logging()


def hydra_main(overrides: List[Any]):
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


if __name__ == "__main__":
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    hydra_main(overrides=overrides)
