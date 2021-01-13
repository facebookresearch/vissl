# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import sys
from argparse import Namespace
from typing import Any, List

import torch
from hydra.experimental import compose, initialize_config_module
from run_distributed_engines import launch_distributed
from torch import nn
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
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import merge_features


def nearest_neighbor_test(cfg: AttrDict, layer_name: str = "heads"):
    temperature = cfg.NEAREST_NEIGHBOR.SIGMA
    num_neighbors = cfg.NEAREST_NEIGHBOR.TOPK
    output_dir = get_checkpoint_folder(cfg)
    logging.info(f"Testing with sigma: {temperature}, topk neighbors: {num_neighbors}")

    ############################################################################
    # Step 1: get train and test features
    train_out = merge_features(output_dir, "train", layer_name, cfg)
    train_features, train_labels = train_out["features"], train_out["targets"]
    # put train features and labels on gpu and transpose train features
    train_features = torch.from_numpy(train_features).float().cuda().t()
    train_labels = torch.LongTensor(train_labels).cuda()
    num_classes = train_labels.max() + 1
    if cfg.NEAREST_NEIGHBOR.L2_NORM_FEATS:
        train_features = nn.functional.normalize(train_features, dim=0, p=2)

    test_out = merge_features(output_dir, "test", layer_name, cfg)
    test_features, test_labels = test_out["features"], test_out["targets"]

    ###########################################################################
    # Step 2: calculate the nearest neighbor and the metrics
    top1, top5, total = 0.0, 0.0, 0
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(num_neighbors, num_classes).cuda()
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images and normalize the features if needed
            features = test_features[
                idx : min((idx + imgs_per_chunk), num_test_images), :
            ]
            targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images), :]
            batch_size = targets.shape[0]
            features = torch.from_numpy(features).float().cuda()
            targets = torch.LongTensor(targets).cuda()
            if cfg.NEAREST_NEIGHBOR.L2_NORM_FEATS:
                features = nn.functional.normalize(features, dim=1, p=2)

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features)
            distances, indices = similarity.topk(
                num_neighbors, largest=True, sorted=True
            )
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * num_neighbors, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
            distances_transform = distances.clone().div_(temperature).exp_()
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()
            total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    logging.info(f"Total images: {total}, Top1: {top1}, Top5: {top5}")
    return top1, top5


def main(args: Namespace, config: AttrDict):
    # setup logging
    setup_logging(__name__)

    # print the coniguration used
    print_cfg(config)

    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=config)

    # extract the features
    launch_distributed(
        config,
        args.node_id,
        engine_name="extract_features",
        hook_generator=default_hook_generator,
    )

    # Get the names of the features that we are extracting. If user doesn't
    # specify the features to evaluate, we get the full model output and freeze
    # head/trunk both as caution.
    feat_names = get_trunk_output_feature_names(config.MODEL)
    if len(feat_names) == 0:
        feat_names = ["heads"]

    for layer in feat_names:
        top1, top5 = nearest_neighbor_test(config, layer_name=layer)
        logging.info(f"layer: {layer} Top1: {top1}, Top5: {top5}")
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
