# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script provides capability to cluster features into certain number clusters
using FAISS and assigning the hard labels to the dataset.
"""

import logging
import sys
from argparse import Namespace
from typing import Any, List

import faiss
import numpy as np
from hydra.experimental import compose, initialize_config_module
from run_distributed_engines import launch_distributed
from vissl.data import build_dataset
from vissl.hooks import default_hook_generator
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import AttrDict, convert_to_attrdict, is_hydra_available
from vissl.utils.io import save_file
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import merge_features, set_seeds


def get_data_features_and_images(cfg: AttrDict):
    output_dir = get_checkpoint_folder(cfg)
    split = cfg.CLUSTERFIT.FEATURES.DATA_PARTITION
    logging.info("Merging features...")
    # merge the features across all nodes/gpus into one
    feature_data = merge_features(
        output_dir, split.lower(), cfg.CLUSTERFIT.FEATURES.LAYER_NAME, cfg
    )

    logging.info("Getting the image paths...")
    # get the list of image Ids
    dataset = build_dataset(cfg, split)
    feature_image_paths = dataset.get_image_paths()
    # due to multi-modality, we get image_paths as a nested list, one for each
    # dataset. Check it's a list and extract images.
    assert type(feature_image_paths) == list, "Image paths must be a list"
    assert len(feature_image_paths) == 1, "Multi-modality not supported yet!"
    return feature_data, feature_image_paths[0]


def cluster_features_and_label(args: Namespace, cfg: AttrDict):
    cluster_backend = cfg.CLUSTERFIT.CLUSTER_BACKEND
    num_clusters = cfg.CLUSTERFIT.NUM_CLUSTERS
    data_split = cfg.CLUSTERFIT.FEATURES.DATA_PARTITION
    data_name = cfg.CLUSTERFIT.FEATURES.DATASET_NAME
    n_iter = cfg.CLUSTERFIT.N_ITER
    output_dir = get_checkpoint_folder(cfg)

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
    cluster_output_filepath = (
        f"{output_dir}/{data_name}_{data_split}_N{num_clusters}_{cluster_backend}.pkl"
    )
    hard_labels_output_filepath = (
        f"{output_dir}/"
        f"{data_name}_{data_split}_N{num_clusters}_{cluster_backend}_lbls.npy"
    )
    out_hard_labels = np.array(hard_cluster_labels.tolist(), dtype=np.int64).reshape(-1)
    save_file(clustering_output_dict, cluster_output_filepath)
    save_file(out_hard_labels, hard_labels_output_filepath)
    logging.info("All Done!")


def main(args: Namespace, cfg: AttrDict):
    # setup logging
    setup_logging(__name__)

    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=cfg)

    # set seeds
    logging.info("Setting seed....")
    set_seeds(cfg, args.node_id)

    # extract the features. We enable the feature extraction as well.
    launch_distributed(
        cfg,
        args.node_id,
        engine_name="extract_features",
        hook_generator=default_hook_generator,
    )

    # cluster the extracted features
    cluster_features_and_label(args, cfg)
    # close the logging streams including the filehandlers
    shutdown_logging()


def hydra_main(overrides: List[Any]):
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


if __name__ == "__main__":
    """
    Example usage:

    `python tools/cluster_features_and_label.py \
        config=pretrain/clusterfit/cluster_features_resnet_8gpu_rotation_in1k`
    """
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    hydra_main(overrides=overrides)
