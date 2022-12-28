# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, NamedTuple

import numpy as np
import torch
from torch import nn
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
from vissl.utils.io import save_file
from vissl.utils.logger import setup_logging, shutdown_logging


class MaxSimilarityPriorityQueue:
    def __init__(self, max_size: int):
        self.similarities = None
        self.targets = None
        self.max_size = max_size

    def push_all(self, similarities: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Push distances and associated targets to the min priority queue,
        popping all the smallest entries to keep the queue below its max size
        """
        assert similarities.shape[0] == targets.shape[0]

        if self.similarities is None:
            self.similarities = similarities.cpu()
            self.targets = targets.cpu()
        else:
            self.similarities = torch.cat(
                [self.similarities, similarities.cpu()], dim=1
            )
            self.targets = torch.cat([self.targets, targets.cpu()], dim=1)

        if self.similarities.shape[0] > self.max_size:
            self.similarities, indices = self.similarities.topk(
                self.max_size, largest=True, sorted=True
            )
            self.targets = torch.gather(self.targets, dim=1, index=indices)
            assert self.targets.shape == self.similarities.shape

    def pop_all(self):
        """
        Return the largest similarities and associated targets
        """
        return self.similarities, self.targets


class Accuracies(NamedTuple):
    """
    Helper class to compute the top_1 and top_5 accuracies
    """

    correct_top_1: float = 0.0
    correct_top_5: float = 0.0
    total: int = 0

    def __add__(self, other: "Accuracies") -> "Accuracies":
        return Accuracies(
            correct_top_1=self.correct_top_1 + other.correct_top_1,
            correct_top_5=self.correct_top_5 + other.correct_top_5,
            total=self.total + other.total,
        )

    @classmethod
    def from_batch(
        cls, predictions: torch.Tensor, targets: torch.Tensor
    ) -> "Accuracies":
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = correct.narrow(dim=1, start=0, length=1).sum().item()
        if correct.shape[1] >= 5:
            top5 = correct.narrow(dim=1, start=0, length=5).sum().item()
        else:
            top5 = 0.0
        return Accuracies(correct_top_1=top1, correct_top_5=top5, total=targets.size(0))

    @property
    def top_1(self):
        return self.correct_top_1 * 100.0 / self.total

    @property
    def top_5(self):
        return self.correct_top_5 * 100.0 / self.total

    def log(self, layer_name: str):
        logging.info(
            f"Total images({layer_name}): {self.total}, Top1: {self.top_1}, Top5: {self.top_5}"
        )


@torch.no_grad()
def run_knn_at_layer_low_memory(cfg: AttrDict, layer_name: str = "heads"):
    """
    Alternate implementation of kNN which scales to bigger features
    and bigger "train" splits
    """
    if cfg.NEAREST_NEIGHBOR.USE_CUDA:
        logging.warning(
            "config.NEAREST_NEIGHBOR.USE_CUDA is not available when "
            "config.NEAREST_NEIGHBOR.OPTIMIZE_MEMORY is set to True, "
            "using CPU instead"
        )

    temperature = cfg.NEAREST_NEIGHBOR.SIGMA
    num_neighbors = cfg.NEAREST_NEIGHBOR.TOPK
    feature_dir = cfg.NEAREST_NEIGHBOR.FEATURES.PATH
    output_dir = get_checkpoint_folder(cfg)
    logging.info(f"Testing with sigma: {temperature}, topk neighbors: {num_neighbors}")

    # Step 1: get the test features (the train features might not feat in memory)
    test_out = ExtractedFeaturesLoader.load_features(
        feature_dir, "test", layer_name, flatten_features=True
    )
    test_features, test_labels = test_out["features"], test_out["targets"]
    test_features = torch.from_numpy(test_features).float()
    test_feature_num = test_features.shape[0]

    # Step 2: normalize the features if needed
    if cfg.NEAREST_NEIGHBOR.L2_NORM_FEATS:
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    # Step 3: collect the similarity score of each test feature
    # to all the train features, making sure:
    # - never to load the all train features at once to avoid OOM
    # - to keep just the 'num_neighbors' best similarity scores
    shard_paths = ExtractedFeaturesLoader.get_shard_file_names(
        input_dir=feature_dir, split="train", layer=layer_name
    )
    similarity_queue = MaxSimilarityPriorityQueue(max_size=num_neighbors)
    num_classes = 0
    for shard_path in shard_paths:
        shard_content = ExtractedFeaturesLoader.load_feature_shard(shard_path)
        train_features = torch.from_numpy(shard_content.features)
        train_features = train_features.float().reshape((train_features.shape[0], -1))
        if cfg.NEAREST_NEIGHBOR.L2_NORM_FEATS:
            train_features = nn.functional.normalize(train_features, dim=1, p=2)
        train_features = train_features.t()

        train_labels = torch.LongTensor(shard_content.targets).squeeze(-1)
        num_classes = max(num_classes, train_labels.max().item() + 1)
        similarities = torch.mm(test_features, train_features)
        if similarities.shape[0] > num_neighbors:
            distances, indices = similarities.topk(
                num_neighbors, largest=True, sorted=True
            )
        else:
            distances, indices = torch.sort(similarities, descending=True)
        closest_labels = train_labels[indices]
        similarity_queue.push_all(distances, closest_labels)

    # Step 4: collect the samples with the closest similarities
    # for each test sample, and assemble it in a matrix with
    # shape (num_test_samples, num_neighbors)
    topk_distances, topk_labels = similarity_queue.pop_all()

    # Step 5: go through each of the test samples, batch by batch,
    # to compute the label of each test sample based on the top k
    # nearest neighbors and their corresponding labels
    accuracies = Accuracies()
    output_targets, output_predicted_label, output_inds = [], [], []

    batch_size = 100
    num_test_images = test_feature_num
    for idx in range(0, num_test_images, batch_size):
        min_idx = idx
        max_idx = min(idx + batch_size, num_test_images)

        distances = topk_distances[min_idx:max_idx, ...]
        retrieved_neighbors = topk_labels[min_idx:max_idx, ...]
        targets = torch.LongTensor(test_labels[min_idx:max_idx])

        retrieval_one_hot = torch.zeros(batch_size * num_neighbors, num_classes)
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        predictions = get_sorted_predictions(
            batch_size, num_classes, distances, retrieval_one_hot, temperature
        )

        # find the predictions that match the target
        accuracies = accuracies + Accuracies.from_batch(predictions, targets)

        # get the predictions, nearest neighbors, inds to save
        output_inds.extend(range(min_idx, max_idx))
        output_predicted_label.append(predictions.data.cpu().numpy())
        output_targets.append(targets.data.cpu().numpy())

    _save_knn_results(
        output_dir, layer_name, output_inds, output_predicted_label, output_targets
    )
    accuracies.log(layer_name)
    return accuracies.top_1, accuracies.top_5, accuracies.total


def run_knn_at_layer(cfg: AttrDict, layer_name: str = "heads"):
    """
    Run the Nearest Neighbour benchmark at the layer "layer_name"
    """
    temperature = cfg.NEAREST_NEIGHBOR.SIGMA
    num_neighbors = cfg.NEAREST_NEIGHBOR.TOPK
    feature_dir = cfg.NEAREST_NEIGHBOR.FEATURES.PATH
    output_dir = get_checkpoint_folder(cfg)
    logging.info(f"Testing with sigma: {temperature}, topk neighbors: {num_neighbors}")

    ############################################################################
    # Step 1: get train and test features
    train_out = ExtractedFeaturesLoader.load_features(
        feature_dir, "train", layer_name, flatten_features=True
    )
    train_features, train_labels = train_out["features"], train_out["targets"]
    test_out = ExtractedFeaturesLoader.load_features(
        feature_dir, "test", layer_name, flatten_features=True
    )
    test_features, test_labels = test_out["features"], test_out["targets"]
    train_features = torch.from_numpy(train_features).float()
    test_features = torch.from_numpy(test_features).float()
    train_labels = torch.LongTensor(train_labels)
    num_classes = train_labels.max() + 1

    ###########################################################################
    # Step 2: calculate the nearest neighbor and the metrics
    accuracies = Accuracies()
    if cfg.NEAREST_NEIGHBOR.L2_NORM_FEATS:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    # put train features and labels on gpu and transpose train features
    if cfg.NEAREST_NEIGHBOR.USE_CUDA:
        train_features = train_features.cuda().t()
        test_features = test_features.cuda()
        train_labels = train_labels.cuda()
    else:
        train_features = train_features.t()

    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    output_targets, output_predicted_label, output_inds = [], [], []
    with torch.no_grad():
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images and normalize the features if needed
            features = test_features[
                idx : min((idx + imgs_per_chunk), num_test_images), :
            ]
            targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images), :]
            batch_size = targets.shape[0]
            targets = torch.LongTensor(targets)
            if cfg.NEAREST_NEIGHBOR.USE_CUDA:
                targets = torch.LongTensor(targets).cuda()

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features)
            distances, indices = similarity.topk(
                num_neighbors, largest=True, sorted=True
            )
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot = torch.zeros(batch_size * num_neighbors, num_classes)
            if cfg.NEAREST_NEIGHBOR.USE_CUDA:
                retrieval_one_hot = retrieval_one_hot.cuda()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            predictions = get_sorted_predictions(
                batch_size, num_classes, distances, retrieval_one_hot, temperature
            )

            # find the predictions that match the target
            accuracies = accuracies + Accuracies.from_batch(predictions, targets)

            # get the predictions, nearest neighbors, inds to save
            output_inds.extend(range(idx, min((idx + imgs_per_chunk), num_test_images)))
            output_predicted_label.append(predictions.data.cpu().numpy())
            output_targets.append(targets.data.cpu().numpy())

    _save_knn_results(
        output_dir, layer_name, output_inds, output_predicted_label, output_targets
    )
    accuracies.log(layer_name)
    return accuracies.top_1, accuracies.top_5, accuracies.total


def get_sorted_predictions(
    batch_size: int,
    num_classes: int,
    distances: torch.Tensor,
    retrieval_one_hot: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    From the closest neighbors compute the top predicted classes
    by blending the probabilities of each neighbor's class
    """
    assert distances.shape[0] == batch_size
    num_neighbors = distances.shape[1]
    assert retrieval_one_hot.shape[0] == batch_size * num_neighbors
    assert retrieval_one_hot.shape[1] == num_classes

    distances = distances.clone().div_(temperature).exp_()
    probs = torch.sum(
        torch.mul(
            retrieval_one_hot.view(batch_size, num_neighbors, num_classes),
            distances.view(batch_size, num_neighbors, 1),
        ),
        dim=1,
    )
    _, predictions = probs.sort(dim=1, descending=True)

    assert probs.shape == torch.Size([batch_size, num_classes])
    assert predictions.shape == torch.Size([batch_size, num_classes])
    return predictions


def _save_knn_results(
    output_dir: str,
    layer_name: str,
    output_inds: List[int],
    output_predicted_label: List[np.ndarray],
    output_targets: List[np.ndarray],
):
    output_targets = np.vstack(output_targets)
    output_predicted_label = np.vstack(output_predicted_label)
    logging.info(
        f"Saving targets: {output_targets.shape}, "
        f"output predictions: {output_predicted_label.shape}, "
        f"output indices: {len(output_inds)}"
    )
    save_file(
        output_predicted_label, f"{output_dir}/kNN_{layer_name}_output_predictions.npy"
    )
    save_file(output_targets, f"{output_dir}/kNN_{layer_name}_output_targets.npy")
    save_file(output_inds, f"{output_dir}/kNN_{layer_name}_output_image_indices.npy")


def run_knn_at_all_layers(config: AttrDict):
    """
    Get the names of the features that we are extracting. If user doesn't
    specify the features to evaluate, we get the full model output and freeze
    head/trunk both as caution.
    """
    feat_names = get_trunk_output_feature_names(config.MODEL)
    if len(feat_names) == 0:
        feat_names = ["heads"]

    top_k_list = config.NEAREST_NEIGHBOR.TOPK
    if not isinstance(top_k_list, list):
        top_k_list = [top_k_list]

    for layer in feat_names:
        # TODO - replace this with more optimal approach:
        #  * use the max top_k to select the closest neighbors
        #  * then only at the end, sub-select the different top_k
        #    to compute the accuracy for each different top_k
        for top_k in top_k_list:
            config.NEAREST_NEIGHBOR.TOPK = top_k
            if config.NEAREST_NEIGHBOR.OPTIMIZE_MEMORY:
                top1, top5, _ = run_knn_at_layer_low_memory(config, layer_name=layer)
            else:
                top1, top5, _ = run_knn_at_layer(config, layer_name=layer)
            logging.info(f"layer: {layer}, topk: {top_k}, Top1: {top1}, Top5: {top5}")


def extract_features_and_run_knn(node_id: int, config: AttrDict):
    setup_logging(__name__)
    print_cfg(config)
    set_env_vars(local_rank=0, node_id=0, cfg=config)

    # Extract the features if no path to the extract features is provided
    if not config.NEAREST_NEIGHBOR.FEATURES.PATH:
        launch_distributed(
            config,
            node_id,
            engine_name="extract_features",
            hook_generator=default_hook_generator,
        )
        config.NEAREST_NEIGHBOR.FEATURES.PATH = get_checkpoint_folder(config)

    # Run KNN on all the extract features
    run_knn_at_all_layers(config)

    # close the logging streams including the file handlers
    shutdown_logging()


class _ResumableNearestNeighborSlurmJob:
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
        extract_features_and_run_knn(node_id=node_id, config=self.config)

    def checkpoint(self):
        import submitit

        trainer = _ResumableNearestNeighborSlurmJob(config=self.config)
        return submitit.helpers.DelayedSubmission(trainer)


def extract_features_and_run_knn_on_slurm(cfg):
    executor = create_submitit_executor(cfg)
    trainer = _ResumableNearestNeighborSlurmJob(config=cfg)
    job = executor.submit(trainer)
    print(f"SUBMITTED: {job.job_id}")
    return job
