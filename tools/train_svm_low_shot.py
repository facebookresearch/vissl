# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing as mp
import sys
from argparse import Namespace
from typing import Any, List

from extra_scripts.create_low_shot_samples import (
    generate_low_shot_samples,
    generate_places_low_shot_samples,
)
from vissl.config import AttrDict
from vissl.data import dataset_catalog
from vissl.hooks import default_hook_generator
from vissl.models.model_helpers import get_trunk_output_feature_names
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import (
    compose_hydra_configuration,
    convert_to_attrdict,
    print_cfg,
)
from vissl.utils.io import load_file
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import merge_features
from vissl.utils.svm_utils.svm_low_shot_trainer import SVMLowShotTrainer


def train_voc07_low_shot(
    k_values: List[int],
    sample_inds: List[int],
    output_dir: str,
    layername: str,
    cfg: AttrDict,
):
    dataset_name = cfg["SVM"]["low_shot"]["dataset_name"]
    low_shot_trainer = SVMLowShotTrainer(
        cfg["SVM"], layer=layername, output_dir=output_dir
    )
    train_data = merge_features(output_dir, "train", layername)
    train_features, train_targets = train_data["features"], train_data["targets"]
    test_data = merge_features(output_dir, "test", layername)
    test_features, test_targets = test_data["features"], test_data["targets"]
    # now we want to create the low-shot samples based on the kind of dataset.
    # We only create low-shot samples for training. We test on the full dataset.
    generate_low_shot_samples(
        dataset_name, train_targets, k_values, sample_inds, output_dir, layername
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
    results = low_shot_trainer.aggregate_stats(k_values, sample_inds)
    logging.info("All Done!")
    return results


def train_sample_places_low_shot(
    low_shot_trainer: SVMLowShotTrainer,
    k_values: List[int],
    sample_inds: List[int],
    sample_num: int,
    output_dir: str,
    layername: str,
    cfg: AttrDict,
):
    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=cfg)

    for low_shot_kvalue in k_values:
        checkpoint_dir = f"{output_dir}/sample{sample_num}_k{low_shot_kvalue}"
        train_data = merge_features(checkpoint_dir, "train", layername)
        train_features = train_data["features"]
        train_targets = train_data["targets"]
        checkpoint_dir = f"{output_dir}/sample{sample_inds[0]}_k{k_values[0]}"
        test_data = merge_features(checkpoint_dir, "test", layername)
        test_features = test_data["features"]
        test_targets = test_data["targets"]
        low_shot_trainer.train(
            train_features, train_targets, sample_num, low_shot_kvalue
        )
        low_shot_trainer.test(test_features, test_targets, sample_num, low_shot_kvalue)


def train_places_low_shot(
    k_values: List[int],
    sample_inds: List[int],
    output_dir: str,
    layername: str,
    cfg: AttrDict,
):
    low_shot_trainer = SVMLowShotTrainer(
        cfg["SVM"], layer=layername, output_dir=output_dir
    )

    # we have extracted the features in the
    running_tasks = [
        mp.Process(
            target=train_sample_places_low_shot,
            args=(
                low_shot_trainer,
                k_values,
                sample_inds,
                sample_num,
                output_dir,
                layername,
                cfg,
            ),
        )
        for sample_num in sample_inds
    ]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()
    results = low_shot_trainer.aggregate_stats(k_values, sample_inds)
    logging.info(f"All Done for layer: {layername}")
    return results


def train_svm_low_shot(cfg: AttrDict, output_dir: str, layername: str):
    logging.info(f"Training Low-shot SVM for layer: {layername}")
    k_values = cfg["SVM"]["low_shot"]["k_values"]
    sample_inds = cfg["SVM"]["low_shot"]["sample_inds"]
    dataset_name = cfg["SVM"]["low_shot"]["dataset_name"]

    if "voc" in dataset_name:
        results = train_voc07_low_shot(
            k_values, sample_inds, output_dir, layername, cfg
        )
    elif "places" in dataset_name:
        results = train_places_low_shot(
            k_values, sample_inds, output_dir, layername, cfg
        )
    return results


def extract_low_shot_features(args: Namespace, cfg: AttrDict, output_dir: str):
    dataset_name = cfg["SVM"]["low_shot"]["dataset_name"]
    k_values = cfg["SVM"]["low_shot"]["k_values"]
    sample_inds = cfg["SVM"]["low_shot"]["sample_inds"]
    if "voc" in dataset_name:
        # extract the features. In case of voc07 low-shot, we extract the
        # features on full train and test sets. Both sets have about 5K images
        # we extract
        launch_distributed(
            cfg,
            args.node_id,
            engine_name="extract_features",
            hook_generator=default_hook_generator,
        )
    elif "places" in dataset_name:
        # in case of places, since the features size could become large, we need
        # to extract features at smaller subsamples
        data_paths, label_paths = dataset_catalog.get_data_files(
            split="TRAIN", dataset_config=cfg["DATA"]
        )
        targets = load_file(label_paths[0])
        logging.info("Generating low-shot samples for Places205...")
        generate_places_low_shot_samples(
            targets, k_values, sample_inds, output_dir, data_paths[0]
        )

        test_features_extracted = False
        for idx in sample_inds:
            for k in k_values:
                out_img_file = f"{output_dir}/train_images_sample{idx}_k{k}.npy"
                out_lbls_file = f"{output_dir}/train_labels_sample{idx}_k{k}.npy"
                cfg.DATA.TRAIN.DATA_PATHS = [out_img_file]
                cfg.DATA.TRAIN.LABEL_PATHS = [out_lbls_file]
                cfg.CHECKPOINT.DIR = f"{output_dir}/sample{idx}_k{k}"
                logging.info(
                    f"Extracting features for places low shot: sample{idx}_k{k}"
                )
                # we want to extract the test features only once since the test
                # features are commonly used for testing for all low-shot setup.
                if test_features_extracted:
                    cfg.TEST_MODEL = False
                launch_distributed(
                    cfg,
                    args.node_id,
                    engine_name="extract_features",
                    hook_generator=default_hook_generator,
                )
                test_features_extracted = True
        # set the test model to true again after feature extraction is done
        cfg.TEST_MODEL = True
    else:
        raise RuntimeError(f"Dataset not recognised: {dataset_name}")


def main(args: Namespace, cfg: AttrDict):
    # setup logging
    setup_logging(__name__)

    # print the cfg
    print_cfg(cfg)

    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=cfg)

    output_dir = get_checkpoint_folder(cfg)

    assert cfg.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON, (
        "Feature eval mode is not ON. Can't run train_svm. "
        "Set config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True "
        "in your config or from command line."
    )
    extract_low_shot_features(args, cfg, output_dir)

    # Get the names of the features that we extracted features for. If user doesn't
    # specify the features to evaluate, we get the full model output and freeze
    # head/trunk both as caution.
    layers = get_trunk_output_feature_names(cfg.MODEL)
    if len(layers) == 0:
        layers = ["heads"]

    # train low shot svm for each layer.
    output = {}
    for layer in layers:
        results = train_svm_low_shot(cfg, output_dir, layer)
        output[layer] = results
    logging.info(f"Results: {output}")

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
