# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import sys

import numpy as np
from extra_scripts.create_voc07_low_shot_samples import generate_voc07_low_shot_samples
from hydra.experimental import compose, initialize_config_module
from vissl.utils.checkpoint import get_absolute_path
from vissl.utils.hydra_config import convert_to_attrdict, is_hydra_available, print_cfg
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import merge_features
from vissl.utils.svm_utils.svm_low_shot_trainer import SVMLowShotTrainer


def train_svm_low_shot(args, cfg):
    # print the cfg
    print_cfg(cfg)

    output_dir = get_absolute_path(cfg.SVM.OUTPUT_DIR)
    logging.info(f"Training Low-shot SVM for layer: {args.layername}")
    low_shot_trainer = SVMLowShotTrainer(cfg["SVM"], layer=args.layername)
    train_data = merge_features(output_dir, "train", args.layername, cfg)
    train_features, train_targets = train_data["features"], train_data["targets"]
    test_data = merge_features(output_dir, "test", args.layername, cfg)
    test_features, test_targets = test_data["features"], test_data["targets"]

    # now we want to create the low-shot samples based on the kind of dataset.
    # We only create low-shot samples for training. We test on the full dataset.
    k_values = cfg["SVM"]["low_shot"]["k_values"]
    sample_inds = cfg["SVM"]["low_shot"]["sample_inds"]
    generate_voc07_low_shot_samples(
        train_targets, k_values, sample_inds, output_dir, args.layername
    )

    # Now, we train and test the low-shot SVM for every sample and k-value.
    for sample_num in sample_inds:
        for low_shot_kvalue in k_values:
            train_targets = np.load(
                os.path.join(
                    output_dir,
                    f"{args.layername}_sample{sample_num}_k{low_shot_kvalue}.npy",
                )
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


def main(args, cfg):
    setup_logging(__name__)
    train_svm_low_shot(args, cfg)


def hydra_main(overrides):
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


if __name__ == "__main__":
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    hydra_main(overrides=overrides)
