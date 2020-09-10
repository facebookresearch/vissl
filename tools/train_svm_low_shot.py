# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import sys
from argparse import Namespace
from typing import Any, List

from extra_scripts.create_voc07_low_shot_samples import generate_voc07_low_shot_samples
from hydra.experimental import compose, initialize_config_module
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
from vissl.utils.svm_utils.svm_low_shot_trainer import SVMLowShotTrainer


def train_svm_low_shot(cfg: AttrDict, output_dir: str, layername: str):
    logging.info(f"Training Low-shot SVM for layer: {layername}")
    low_shot_trainer = SVMLowShotTrainer(
        cfg["SVM"], layer=layername, output_dir=output_dir
    )
    train_data = merge_features(output_dir, "train", layername, cfg)
    train_features, train_targets = train_data["features"], train_data["targets"]
    test_data = merge_features(output_dir, "test", layername, cfg)
    test_features, test_targets = test_data["features"], test_data["targets"]

    # now we want to create the low-shot samples based on the kind of dataset.
    # We only create low-shot samples for training. We test on the full dataset.
    k_values = cfg["SVM"]["low_shot"]["k_values"]
    sample_inds = cfg["SVM"]["low_shot"]["sample_inds"]
    generate_voc07_low_shot_samples(
        train_targets, k_values, sample_inds, output_dir, layername
    )

    # Now, we train and test the low-shot SVM for every sample and k-value.
    for sample_num in sample_inds:
        for low_shot_kvalue in k_values:
            train_targets = load_file(
                f"{output_dir}/{layername}_sample{sample_num}_k{low_shot_kvalue}.npy"
            )
            low_shot_trainer.train(
                train_features, train_targets, sample_num, low_shot_kvalue
            )
            low_shot_trainer.test(
                test_features, test_targets, sample_num, low_shot_kvalue
            )

    # now we aggregate the stats across all independent samples and for each
    # k-value and report mean/min/max/std stats
    low_shot_trainer.aggregate_stats(k_values, sample_inds)
    logging.info("All Done!")
    # close the logging streams including the filehandlers
    shutdown_logging()


def main(args: Namespace, cfg: AttrDict):
    # setup logging
    setup_logging(__name__)

    # print the cfg
    print_cfg(cfg)

    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=cfg)

    # get the layers for which we will train low shot svm
    layers = [
        item[0]
        for item in cfg.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP
    ]
    output_dir = get_checkpoint_folder(cfg)

    # train low shot svm for each layer.
    for layer in layers:
        train_svm_low_shot(cfg, output_dir, layer)


def hydra_main(overrides: List[Any]):
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


if __name__ == "__main__":
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    hydra_main(overrides=overrides)
