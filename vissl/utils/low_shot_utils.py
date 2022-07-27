# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from vissl.config import AttrDict
from vissl.hooks import default_hook_generator
from vissl.models.model_helpers import get_trunk_output_feature_names
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.distributed_launcher import (
    create_submitit_executor,
    launch_distributed,
)
from vissl.utils.env import set_env_vars
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.utils.hydra_config import print_cfg
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.svm_utils.svm_trainer import SVMTrainer


def load_features(feature_dir: str, layer_name: str):
    train_out = ExtractedFeaturesLoader.load_features(
        feature_dir, "train", layer_name, flatten_features=True
    )
    train_features, train_labels = train_out["features"], train_out["targets"]
    test_out = ExtractedFeaturesLoader.load_features(
        feature_dir, "test", layer_name, flatten_features=True
    )
    test_features, test_labels = test_out["features"], test_out["targets"]
    return train_features, train_labels, test_features, test_labels


def run_low_shot_logistic_regression(config: AttrDict, layer_name: str = "heads"):
    """
    Run the Nearest Neighbour benchmark at the layer "layer_name"
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # -- get train and test features
    feature_dir = config.LOW_SHOT_BENCHMARK.FEATURES.PATH
    train_features, train_labels, test_features, test_labels = load_features(
        feature_dir, layer_name
    )

    # -- Scale the features based on training statistics
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    # -- Fit Logistic Regression Classifier
    method_config = config.LOW_SHOT_BENCHMARK.LOGISTIC_REGRESSION
    lambd = method_config.LAMBDA
    lambd /= len(train_labels)
    classifier = LogisticRegression(
        penalty="l2",
        C=1 / lambd,
        multi_class="multinomial",
    )
    classifier.fit(
        train_features,
        train_labels,
    )

    # -- Evaluate on train and test set
    train_score = classifier.score(train_features, train_labels)
    print("Train score: ", train_score)
    test_score = classifier.score(test_features, test_labels)
    print("Test score: ", test_score)
    return test_score


def run_low_shot_svm(config: AttrDict, layer_name: str = "heads"):
    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=config)

    features_dir = config.LOW_SHOT_BENCHMARK.FEATURES.PATH
    output_dir = get_checkpoint_folder(config)

    # Train the svm
    logging.info(f"Training SVM for layer: {layer_name}")
    trainer = SVMTrainer(config.SVM, layer=layer_name, output_dir=output_dir)
    train_data = ExtractedFeaturesLoader.load_features(
        features_dir, "train", layer_name, flatten_features=True
    )
    trainer.train(train_data["features"], train_data["targets"])

    # Test the svm
    test_data = ExtractedFeaturesLoader.load_features(
        features_dir, "test", layer_name, flatten_features=True
    )
    trainer.test(test_data["features"], test_data["targets"])
    logging.info("All Done!")


def run_low_shot_all_layers(config: AttrDict):
    """
    Get the names of the features that we are extracting. If user doesn't
    specify the features to evaluate, we get the full model output and freeze
    head/trunk both as caution.
    """
    feat_names = get_trunk_output_feature_names(config.MODEL)
    if len(feat_names) == 0:
        feat_names = ["heads"]

    for layer in feat_names:
        if config.LOW_SHOT_BENCHMARK.METHOD == "logistic_regression":
            top_1 = run_low_shot_logistic_regression(config, layer_name=layer)
        else:
            top_1 = run_low_shot_svm(config, layer_name=layer)
        logging.info(f"layer: {layer}, Top1: {top_1}")


def extract_features_and_low_shot(node_id: int, config: AttrDict):
    setup_logging(__name__)
    print_cfg(config)
    set_env_vars(local_rank=0, node_id=0, cfg=config)

    # Extract the features if no path to the extract features is provided
    if not config.LOW_SHOT_BENCHMARK.FEATURES.PATH:
        launch_distributed(
            config,
            node_id,
            engine_name="extract_features",
            hook_generator=default_hook_generator,
        )
        config.LOW_SHOT_BENCHMARK.FEATURES.PATH = get_checkpoint_folder(config)

    # Run KNN on all the extract features
    run_low_shot_all_layers(config)

    # close the logging streams including the file handlers
    shutdown_logging()


class _ResumableLowShotSlurmJob:
    def __init__(self, config: AttrDict):
        self.config = config

    def __call__(self):
        import submitit

        environment = submitit.JobEnvironment()
        node_id = environment.global_rank
        master_ip = environment.hostnames[0]
        master_port = self.config.SLURM.PORT_ID
        self.config.DISTRIBUTED.INIT_METHOD = "tcp"
        self.config.DISTRIBUTED.RUN_ID = f"{master_ip}:{master_port}"
        extract_features_and_low_shot(node_id=node_id, config=self.config)

    def checkpoint(self):
        import submitit

        trainer = _ResumableLowShotSlurmJob(config=self.config)
        return submitit.helpers.DelayedSubmission(trainer)


def extract_features_and_low_shot_on_slurm(cfg):
    executor = create_submitit_executor(cfg)
    trainer = _ResumableLowShotSlurmJob(config=cfg)
    job = executor.submit(trainer)
    print(f"SUBMITTED: {job.job_id}")
    return job
