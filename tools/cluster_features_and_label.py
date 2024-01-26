# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script provides capability to cluster features into certain number clusters
using FAISS and assigning the hard labels to the dataset.
"""

import logging
import os
import sys
from argparse import Namespace
from typing import Any, List, Optional

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from vissl.config import AttrDict
from vissl.data import build_dataset
from vissl.hooks import default_hook_generator
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.env import set_env_vars, setup_path_manager
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from vissl.utils.io import save_file
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import is_faiss_available, set_seeds
from vissl.utils.pca import PCA


def get_data_features_for_k_means(cfg: AttrDict):
    """
    Sample the extract features from disk by reading through the
    extracted feature shards and return a sub-set
    """
    return ExtractedFeaturesLoader.sample_features(
        input_dir=cfg.CLUSTERFIT.FEATURES.PATH,
        split=cfg.CLUSTERFIT.FEATURES.DATA_PARTITION.lower(),
        layer=cfg.CLUSTERFIT.FEATURES.LAYER_NAME,
        num_samples=cfg.CLUSTERFIT.DATA_LIMIT,
        seed=cfg.CLUSTERFIT.DATA_LIMIT_SAMPLING.SEED,
        flatten_features=True,
    )


def get_image_paths(cfg: AttrDict, split: str) -> List[str]:
    """
    Get the list of image path for the provided dataset and split
    """
    dataset = build_dataset(cfg=cfg, split=split)
    feature_image_paths = dataset.get_image_paths()
    # due to multi-modality, we get image_paths as a nested list, one for each
    # dataset. Check it's a list and extract images.
    assert type(feature_image_paths) == list, "Image paths must be a list"
    assert len(feature_image_paths) == 1, "Multi-modality not supported yet!"
    return feature_image_paths[0]


def cluster_features(cfg: AttrDict):
    assert is_faiss_available(), (
        "Please install faiss using conda install faiss-gpu -c pytorch "
        "if using conda or pip install faiss-gpu"
    )
    import faiss

    num_clusters = cfg.CLUSTERFIT.NUM_CLUSTERS
    cluster_backend = cfg.CLUSTERFIT.CLUSTER_BACKEND
    data_split = cfg.CLUSTERFIT.FEATURES.DATA_PARTITION

    # Step 1: get a sub-sample of the extract features on the whole dataset
    # in order to compute the centroids
    feature_data = get_data_features_for_k_means(cfg)
    features = feature_data["features"]
    assert features.ndim == 2, f"Invalid feature shape: {features.shape}"
    assert features.dtype == np.float32, "Features are not float32 type"
    logging.info(f"Loaded features: {features.shape}")

    # Step 2: normalize the features and apply dimensionality reduction
    logging.info("Normalizing the features...")
    feat_norm = np.linalg.norm(features, axis=1) + 1e-5
    features = features / feat_norm[:, np.newaxis]
    with_dimensionality_reduction = cfg.CLUSTERFIT.FEATURES.DIMENSIONALITY_REDUCTION > 0
    if with_dimensionality_reduction:
        pca = PCA(n_components=cfg.CLUSTERFIT.FEATURES.DIMENSIONALITY_REDUCTION)
        features = pca.fit_transform(features)
        features = np.ascontiguousarray(features)
        features_dim = cfg.CLUSTERFIT.FEATURES.DIMENSIONALITY_REDUCTION
    else:
        pca = None
        features_dim = features.shape[1]

    # Step 3: compute the centroids for the sub-sampled features
    logging.info(
        f"Clustering {features.shape[0]} features in {num_clusters} clusters..."
    )
    assert cluster_backend == "faiss", "Only faiss clustering is supported currently"
    use_gpu = torch.cuda.device_count() > 0
    num_iter = cfg.CLUSTERFIT.NUM_ITER
    kmeans = faiss.Kmeans(
        features.shape[1], num_clusters, niter=num_iter, verbose=True, gpu=use_gpu
    )
    kmeans.train(features)

    # Step 4: compute the cluster assignment for each of the features of the dataset
    # by streaming through the features (to avoid OOM) and save clustering data
    # and hard cluster labels for the images
    _create_dataset_split(cfg, data_split, features_dim, kmeans, pca)
    if cfg.CLUSTERFIT.FEATURES.TEST_PARTITION:
        test_split = cfg.CLUSTERFIT.FEATURES.TEST_PARTITION
        _create_dataset_split(cfg, test_split, features_dim, kmeans, pca)
    logging.info("All Done!")


def _create_dataset_split(
    cfg: AttrDict, data_split: str, features_dim: int, kmeans, pca: Optional[PCA] = None
):
    """
    Scan the dataset split and create a new classification dataset out of it
    where each image is associated to the centroid the closest in feature space.
    """
    num_clusters = cfg.CLUSTERFIT.NUM_CLUSTERS
    data_name = cfg.CLUSTERFIT.FEATURES.DATASET_NAME
    layer_name = cfg.CLUSTERFIT.FEATURES.LAYER_NAME

    logging.info(
        f"Computing cluster label assignment for each sample in {data_split}..."
    )
    indices = []
    distances = []
    target_clusters = []
    shard_paths = ExtractedFeaturesLoader.get_shard_file_names(
        input_dir=cfg.CLUSTERFIT.FEATURES.PATH,
        split=data_split.lower(),
        layer=cfg.CLUSTERFIT.FEATURES.LAYER_NAME,
    )
    for shard_path in shard_paths:
        shard_content = ExtractedFeaturesLoader.load_feature_shard(shard_path)
        shard_features = shard_content.features

        # TODO - factorize this with above??? normalization at least???
        # Reshape and normalize the loaded features
        shard_features = shard_features.reshape(shard_features.shape[0], -1)
        shard_features_norm = np.linalg.norm(shard_features, axis=1) + 1e-5
        shard_features = shard_features / shard_features_norm[:, np.newaxis]

        if pca is not None:
            shard_features = pca.transform(shard_features)
            shard_features = np.ascontiguousarray(shard_features)
        shard_distances, shard_cluster_labels = kmeans.index.search(shard_features, 1)
        indices.extend(shard_content.indices)
        distances.extend(shard_distances)
        target_clusters.extend(shard_cluster_labels)

    # Step 5: save clustering data and hard cluster labels for the images
    logging.info("Saving centroids and cluster assignments to file...")
    dataset_image_paths = get_image_paths(cfg, split=data_split)
    image_paths = [dataset_image_paths[i] for i in indices]
    data_split = data_split.lower()
    clustering_output_dict = {
        "sample_indices": indices,
        "hard_labels": target_clusters,
        "centroids": kmeans.centroids,
        "distances": distances,
        "images": image_paths,
    }
    output_dir = cfg.CLUSTERFIT.OUTPUT_DIR
    g_pathmgr.mkdirs(output_dir)
    output_prefix = (
        f"{data_name}_{data_split}_{layer_name}_N{num_clusters}_D{features_dim}"
    )
    cluster_output_filepath = os.path.join(output_dir, f"{output_prefix}.pkl")
    labels_output_filepath = os.path.join(output_dir, f"{output_prefix}_labels.npy")
    image_path_filepath = os.path.join(output_dir, f"{output_prefix}_images.npy")
    out_images = np.array(image_paths)
    out_hard_labels = np.array(target_clusters, dtype=np.int64).reshape(-1)
    save_file(clustering_output_dict, cluster_output_filepath)
    save_file(out_images, image_path_filepath)
    save_file(out_hard_labels, labels_output_filepath)


def main(args: Namespace, cfg: AttrDict):
    setup_logging(__name__, output_dir=get_checkpoint_folder(cfg))

    # Extract the features if the feature extract is enabled
    if cfg.CLUSTERFIT.FEATURES.EXTRACT:

        # We cannot have automatic extraction with more than 1 node or otherwise
        # we would have to run this script on several nodes and thus have several
        # parallel clustering of the features. The automatic extraction is only
        # there as a shortcut when running on a single node
        assert (
            cfg.DISTRIBUTED.NUM_NODES == 1
        ), "Automatic extraction can only work with 1 node"

        # Make sure to dump the features at the desired path
        cfg.CHECKPOINT.DIR = cfg.CLUSTERFIT.FEATURES.PATH
        cfg.CHECKPOINT.APPEND_DISTR_RUN_ID = False

        # Run the extraction of features
        set_env_vars(local_rank=0, node_id=0, cfg=cfg)
        logging.info("Setting seed....")
        set_seeds(cfg, args.node_id)
        launch_distributed(
            cfg,
            args.node_id,
            engine_name="extract_features",
            hook_generator=default_hook_generator,
        )

    # Else setup the path manager (done in set_env_vars) in
    # case of feature extraction above
    else:
        setup_path_manager()

    cluster_features(cfg)
    shutdown_logging()


def hydra_main(overrides: List[Any]):
    cfg = compose_hydra_configuration(overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


def invoke_main() -> None:
    """
    Example usage:

    ```
    python tools/cluster_features_and_label.py
        config=pretrain/clusterfit/clusterfit_resnet_8gpu_imagenet
        config.CLUSTERFIT.FEATURES.PATH=/path/to/extracted/features
        config.CLUSTERFIT.FEATURES.LAYER_NAME=heads
        config.CLUSTERFIT.FEATURES.DATA_PARTITION=TRAIN
        config.CLUSTERFIT.FEATURES.TEST_PARTITION=TEST
        config.CLUSTERFIT.OUTPUT_DIR=/path/to/output/dataset
        config.CLUSTERFIT.NUM_CLUSTERS=160000
        config.CLUSTERFIT.FEATURES.DATASET_NAME=imagenette160
    ```
    """
    overrides = sys.argv[1:]
    hydra_main(overrides=overrides)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
