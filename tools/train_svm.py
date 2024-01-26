# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing as mp
import sys
from argparse import Namespace
from typing import Any, List

import numpy as np
from vissl.config import AttrDict
from vissl.hooks import default_hook_generator
from vissl.models.model_helpers import get_trunk_output_feature_names
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.env import set_env_vars
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.utils.hydra_config import (
    compose_hydra_configuration,
    convert_to_attrdict,
    print_cfg,
)
from vissl.utils.io import load_file
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.svm_utils.svm_trainer import SVMTrainer


def train_svm(cfg: AttrDict, output_dir: str, layername: str):
    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=cfg)
    features_dir = cfg.SVM_FEATURES_PATH

    # train the svm
    logging.info(f"Training SVM for layer: {layername}")
    trainer = SVMTrainer(cfg["SVM"], layer=layername, output_dir=output_dir)
    train_data = ExtractedFeaturesLoader.load_features(
        features_dir, "train", layername, flatten_features=True
    )
    trainer.train(train_data["features"], train_data["targets"])

    # test the svm
    test_data = ExtractedFeaturesLoader.load_features(
        features_dir, "test", layername, flatten_features=True
    )
    trainer.test(test_data["features"], test_data["targets"])
    logging.info("All Done!")


def main(args: Namespace, config: AttrDict):
    # setup logging
    setup_logging(__name__, output_dir=get_checkpoint_folder(config))

    # print the coniguration used
    print_cfg(config)

    assert config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON, (
        "Feature eval mode is not ON. Can't run train_svm. "
        "Set config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True "
        "in your config or from command line."
    )

    # extract the features
    if not config.SVM_FEATURES_PATH:
        launch_distributed(
            config,
            args.node_id,
            engine_name="extract_features",
            hook_generator=default_hook_generator,
        )
        config.SVM_FEATURES_PATH = get_checkpoint_folder(config)

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
    cfg = compose_hydra_configuration(overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


def invoke_main() -> None:
    overrides = sys.argv[1:]
    hydra_main(overrides=overrides)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
