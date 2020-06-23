#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This script provides capability to cluster features into certain number clusters
using FAISS and assigning the hard labels to the dataset.
"""

import logging
import os

import faiss
import hydra
import numpy as np
from omegaconf import DictConfig
from torch.utils.collect_env import get_pretty_env_info
from vissl.dataset import build_dataset
from vissl.utils.checkpoint import get_absolute_path
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.hydra_config import convert_to_attrdict, is_hydra_available, print_cfg
from vissl.utils.io import save_file
from vissl.utils.logger import setup_logging
from vissl.utils.misc import merge_features, set_seeds


def get_data_features_and_images(cfg):
    output_dir = get_absolute_path(cfg.CLUSTERFIT.OUTPUT_DIR)
    split = cfg.CLUSTERFIT.FEATURES.DATA_PARTITION
    logging.info("Merging features...")
    # merge the features across all nodes/gpus into one
    feature_data = merge_features(
        output_dir, split.lower(), cfg.CLUSTERFIT.FEATURES.LAYER_NAME, cfg
    )

    logging.info("Getting the image paths...")
    # get the list of image Ids
    dataset = build_dataset(split)
    feature_image_paths = dataset.get_image_paths()
    # due to multi-modality, we get image_paths as a nested list, one for each
    # dataset. Check it's a list and extract images.
    assert type(feature_image_paths) == list, "Image paths must be a list"
    assert len(feature_image_paths) == 1, "Multi-modality not supported yet!"
    return feature_data, feature_image_paths[0]


def cluster_features_and_label(args, cfg):
    cluster_backend = cfg.CLUSTERFIT.CLUSTER_BACKEND
    num_clusters = cfg.CLUSTERFIT.NUM_CLUSTERS
    data_split = cfg.CLUSTERFIT.FEATURES.DATA_PARTITION
    data_name = cfg.CLUSTERFIT.FEATURES.DATASET_NAME
    n_iter = cfg.CLUSTERFIT.N_ITER
    output_dir = get_absolute_path(cfg.CLUSTERFIT.OUTPUT_DIR)

    ########### Step 1: Extract the features on full dataset ###################
    feature_data, image_paths = get_data_features_and_images(cfg)

    ########### Step 2: Get the data information ###################
    features = feature_data["features"]
    # features are of shape num_samples x feature_dim
    assert features.ndim == 2, f"Features incorrect shape: {features.shape}"
    assert features.dtype == np.float32, "Features are not float32 type"
    logging.info(f"Clustering Features: {features.shape}")

    ########### Step 3: L2 normalize features ###################
    # TODO: we could support PCA here if needed in future.
    logging.info("L2 normalizing the features now...")
    feat_norm = np.linalg.norm(features, axis=1) + 1e-5
    features = features / feat_norm[:, np.newaxis]

    ########### Step 4: Cluster the features ###################
    logging.info("Clustering the features now...")
    assert cluster_backend == "faiss", "Only faiss clustering is supported currently"
    kmeans = faiss.Kmeans(features.shape[1], num_clusters, niter=n_iter, verbose=True)
    kmeans.train(features)
    centroids = kmeans.centroids

    ########### Step 5: Get the cluster assignment for the features ############
    logging.info("Getting cluster label assignment now...")
    distances, hard_cluster_labels = kmeans.index.search(features, 1)

    #### Step 6: Save clustering data and hard cluster labels for the images ###
    data_split = data_split.lower()
    clustering_output_dict = {
        "hard_labels": hard_cluster_labels,
        "centroids": centroids,
        "distances": distances,
    }
    cluster_output_filepath = os.path.join(
        output_dir, f"{data_name}_{data_split}_N{num_clusters}_{cluster_backend}.pkl"
    )
    hard_labels_output_filepath = os.path.join(
        output_dir,
        f"{data_name}_{data_split}_N{num_clusters}_{cluster_backend}_lbls.npy",
    )
    out_hard_labels = np.array(hard_cluster_labels.tolist(), dtype=np.int64).reshape(-1)
    save_file(clustering_output_dict, cluster_output_filepath)
    save_file(out_hard_labels, hard_labels_output_filepath)
    logging.info("All Done!")


def main(args, cfg):
    # setup logging
    setup_logging(__name__)

    # set seeds
    logging.info("Setting seed....")
    set_seeds(cfg)

    # print the training settings and system settings
    local_rank, _ = get_machine_local_and_dist_rank()
    if local_rank == 0:
        print_cfg(cfg)
        logging.info("System config:\n{}".format(get_pretty_env_info()))
    cluster_features_and_label(args, cfg)


@hydra.main(config_path="hydra_configs", config_name="defaults")
def hydra_main(cfg: DictConfig):
    args, config = convert_to_attrdict(cfg)
    main(args, config)


if __name__ == "__main__":
    assert is_hydra_available(), "Make sure to install hydra"
    hydra_main()
