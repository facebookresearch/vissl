# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script provides capability to RANK the features based on nearest
neighbor distance. It uses FAISS to build an Index and then perform
ranking.
"""

import logging
import sys
from argparse import Namespace
from typing import Any, List

import numpy as np
from hydra.experimental import compose, initialize_config_module
from sklearn.decomposition import PCA
from vissl.config import AttrDict
from vissl.data import build_dataset
from vissl.hooks import default_hook_generator
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import convert_to_attrdict, is_hydra_available
from vissl.utils.io import save_file
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.misc import is_faiss_available, merge_features, set_seeds


def get_data_features_and_images(cfg: AttrDict):
    output_dir = get_checkpoint_folder(cfg)
    split = cfg.RANKING.FEATURES.DATA_PARTITION
    logging.info("Merging features...")
    # merge the features across all nodes/gpus into one
    feature_data = merge_features(
        output_dir, split.lower(), cfg.RANKING.FEATURES.LAYER_NAME
    )

    logging.info("Getting the image paths...")
    # get the list of image Ids
    dataset = build_dataset(cfg=cfg, split=split)
    feature_image_paths = dataset.get_image_paths()
    # due to multi-modality, we get image_paths as a nested list, one for each
    # dataset. Check it's a list and extract images.
    assert type(feature_image_paths) == list, "Image paths must be a list"
    assert len(feature_image_paths) == 1, "Multi-modality not supported yet!"
    return feature_data, feature_image_paths[0]


def rank_features(args: Namespace, cfg: AttrDict):
    # faiss is an optional dependency for VISSL.
    assert is_faiss_available(), (
        "Please install faiss using conda install faiss-gpu -c pytorch "
        "if using conda or pip install faiss-gpu"
    )
    import faiss

    ranking_backend = cfg.RANKING.RANKING_BACKEND
    data_split = cfg.RANKING.FEATURES.DATA_PARTITION
    data_name = cfg.RANKING.FEATURES.DATASET_NAME
    output_dir = get_checkpoint_folder(cfg)

    ########### Step 1: Extract the features on full dataset ###################
    feature_data, image_paths = get_data_features_and_images(cfg)

    ########### Step 2: Get the data information ###################
    features = feature_data["features"]
    # features are of shape num_samples x feature_dim
    assert features.ndim == 2, f"Features incorrect shape: {features.shape}"
    assert features.dtype == np.float32, "Features are not float32 type"
    logging.info(f"Ranking Features: {features.shape}")

    ########### Step 3: Optionally L2 normalize features ###################
    if cfg.RANKING.APPLY_PCA:
        logging.info("L2 normalizing the features now...")
        feat_norm = np.linalg.norm(features, axis=1) + 1e-5
        features = features / feat_norm[:, np.newaxis]
        logging.info(f"Projecting down to {cfg.RANKING.PCA_DIM} dims ...")
        features = PCA(n_components=cfg.RANKING.PCA_DIM).fit_transform(features)
        logging.info(f"PCA features: {features.shape}")

    if cfg.RANKING.NORMALIZE_FEATS:
        logging.info("L2 normalizing the features now...")
        feat_norm = np.linalg.norm(features, axis=1) + 1e-5
        features = features / feat_norm[:, np.newaxis]

    ########### Step 4: Build the L2 index on the features ###################
    logging.info(
        "Building the L2 index and searching nearest neighbor with faiss now..."
    )
    assert ranking_backend == "faiss", "Only faiss clustering is supported currently"
    if cfg.RANKING.USE_GPU:
        logging.info("Using gpu for faiss indexing...")
        index = faiss.GpuIndexFlatL2(
            faiss.StandardGpuResources(),
            features.shape[1],
        )
    else:
        logging.info("Using CPU for faiss indexing...")
        index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    logging.info("Doing the nearest neighbor search now...")
    # Num. neighbors here is 2, so for a given point we find that same point at
    # distance 0, and its nearest neighbor
    distances, nn_indices = index.search(features, 2)
    # Remove distance to self, which is always 0
    distances = [d[1] for d in distances]

    ########### Step 5: Sorting the distances now ############
    logging.info("Sorting and ranking based on the L2 distance now...")
    img_paths_and_distances = zip(image_paths, distances)
    img_paths_and_distances = sorted(
        img_paths_and_distances, key=lambda x: x[1], reverse=True
    )
    paths, distances = [x[0] for x in img_paths_and_distances], [
        x[1] for x in img_paths_and_distances
    ]

    #### Step 6: Save image paths and distances... ###
    data_split = data_split.lower()
    ranking_output_dict = {
        "img_paths": paths,
        "distances": distances,
    }
    ranking_output_filepath = (
        f"{output_dir}/ranking_{data_name}_{data_split}_{ranking_backend}.pkl"
    )
    save_file(ranking_output_dict, ranking_output_filepath)
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
    rank_features(args, cfg)
    # close the logging streams including the filehandlers
    shutdown_logging()


def hydra_main(overrides: List[Any]):
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


def invoke_main() -> None:
    """
    Example usage:

    `python tools/extract_feature_and_rank.py \
        config=pretrain/ranking/rank_features_resnet_8gpu_in1k`
    """
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    hydra_main(overrides=overrides)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
