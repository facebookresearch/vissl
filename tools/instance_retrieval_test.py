# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from typing import Any, List

import numpy as np
import torch
import torchvision
from classy_vision.generic.util import copy_model_to_gpu, load_checkpoint
from iopath.common.file_io import g_pathmgr
from vissl.config import AttrDict
from vissl.models import build_model
from vissl.utils.checkpoint import (
    get_checkpoint_folder,
    init_model_from_consolidated_weights,
)
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import (
    compose_hydra_configuration,
    convert_to_attrdict,
    print_cfg,
)
from vissl.utils.instance_retrieval_utils.data_util import (
    CopyDaysDataset,
    GenericInstanceRetrievalDataset,
    InstanceRetrievalDataset,
    InstanceRetrievalImageLoader,
    InstreDataset,
    MultigrainResize,
    RevisitedInstanceRetrievalDataset,
    WhiteningTrainingImageDataset,
    get_average_gem,
    is_copdays_dataset,
    is_instre_dataset,
    is_oxford_paris_dataset,
    is_revisited_dataset,
    is_whiten_dataset,
    l2n,
)
from vissl.utils.instance_retrieval_utils.rmac import get_rmac_descriptors
from vissl.utils.io import load_file, makedir, save_file
from vissl.utils.logger import setup_logging, shutdown_logging
from vissl.utils.pca import load_pca, train_and_save_pca
from vissl.utils.perf_stats import PerfStats, PerfTimer


# frequency at which we log the image number being processed.
LOG_FREQUENCY = 100

# Setup perf timer.
PERF_STATS = PerfStats()


def build_retrieval_model(cfg):
    """
    Builds the model on 1-gpu and initializes from the weight.
    """
    logging.info("Building model....")
    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    if g_pathmgr.exists(cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE):
        init_weights_path = cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE
        logging.info(f"Initializing model from: {init_weights_path}")
        weights = load_checkpoint(init_weights_path, device=torch.device("cuda"))
        skip_layers = cfg.MODEL.WEIGHTS_INIT.get("SKIP_LAYERS", [])
        replace_prefix = cfg.MODEL.WEIGHTS_INIT.get("REMOVE_PREFIX", None)
        append_prefix = cfg.MODEL.WEIGHTS_INIT.get("APPEND_PREFIX", None)
        state_dict_key_name = cfg.MODEL.WEIGHTS_INIT.get("STATE_DICT_KEY_NAME", None)

        init_model_from_consolidated_weights(
            cfg,
            model,
            weights,
            state_dict_key_name=state_dict_key_name,
            skip_layers=skip_layers,
            replace_prefix=replace_prefix,
            append_prefix=append_prefix,
        )
    else:
        # We only throw the warning if not weights file is provided. We want to
        # benchmark the random initialization model too and hence support that.
        logging.warning("Model is randomly initialized....")
    logging.info(f"Model is:\n {model}")
    return model


# Adapted from Dino by Mathilde Caron: https://github.com/facebookresearch/dino/blob/ba9edd18db78a99193005ef991e04d63984b25a8/utils.py#L795 # NOQA
def extract_activation_maps(img, model, img_scalings):
    activation_maps = []

    for scale in img_scalings:
        # Reshape image.
        inp = img.unsqueeze(0)

        # Scale image. If scale == 1, this is a no-op.
        inp = torch.nn.functional.interpolate(
            inp, scale_factor=scale, mode="bilinear", align_corners=False
        )

        vc = inp.cuda()
        feats = model(vc)[0].cpu()
        activation_maps.append(feats)

    return activation_maps


def get_train_features(
    cfg,
    temp_dir,
    train_dataset_name,
    resize_img,
    spatial_levels,
    image_helper,
    train_dataset,
    model,
):
    train_features = []

    def process_train_image(i, out_dir, verbose=False):
        if i % LOG_FREQUENCY == 0:
            logging.info(f"Train Image: {i}"),

        fname_out = None
        if out_dir:
            fname_out = f"{out_dir}/{i}.npy"

        if fname_out and g_pathmgr.exists(fname_out):
            feat = load_file(fname_out)
            train_features.append(feat)
        else:
            with PerfTimer("read_sample", PERF_STATS):
                fname_in = train_dataset.get_filename(i)
                if is_revisited_dataset(train_dataset_name):
                    img = image_helper.load_and_prepare_revisited_image(
                        fname_in, roi=None
                    )
                elif is_whiten_dataset(train_dataset_name):
                    img = image_helper.load_and_prepare_whitening_image(fname_in)
                else:
                    img = image_helper.load_and_prepare_image(fname_in, roi=None)

            with PerfTimer("extract_features", PERF_STATS):
                img_scalings = cfg.IMG_RETRIEVAL.IMG_SCALINGS or [1]
                activation_maps = extract_activation_maps(img, model, img_scalings)

            if verbose:
                print(
                    f"Example train Image raw activation map shape: { activation_maps[0].shape }"  # NOQA
                )

            with PerfTimer("post_process_features", PERF_STATS):
                # once we have the features,
                # we can perform: rmac | gem pooling | l2 norm
                if cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "rmac":
                    descriptors = get_rmac_descriptors(
                        activation_maps[0],
                        spatial_levels,
                        normalize=cfg.IMG_RETRIEVAL.NORMALIZE_FEATURES,
                    )
                elif cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "gem":
                    descriptors = get_average_gem(
                        activation_maps,
                        p=cfg.IMG_RETRIEVAL.GEM_POOL_POWER,
                        add_bias=True,
                    )
                else:
                    descriptors = torch.mean(torch.stack(activation_maps), dim=0)
                    descriptors = descriptors.reshape(descriptors.shape[0], -1)

                # Optionally l2 normalize the features.
                if (
                    cfg.IMG_RETRIEVAL.NORMALIZE_FEATURES
                    and cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE != "rmac"
                ):
                    # RMAC performs normalization within the algorithm, hence we skip it here.
                    descriptors = l2n(descriptors, dim=1)

            if fname_out:
                save_file(descriptors.data.numpy(), fname_out, verbose=False)
            train_features.append(descriptors.data.numpy())

    num_images = train_dataset.get_num_images()

    out_dir = None
    if temp_dir:
        out_dir = f"{temp_dir}/{train_dataset_name}_S{resize_img}_features_train"
        makedir(out_dir)

    logging.info(f"Getting features for train images: {num_images}")
    for i in range(num_images):
        process_train_image(i, out_dir, verbose=(i == 0))

    train_features = np.vstack([x.reshape(-1, x.shape[-1]) for x in train_features])
    logging.info(f"Train features size: {train_features.shape}")
    return train_features


def process_eval_image(
    cfg,
    fname_in,
    roi,
    fname_out,
    spatial_levels,
    image_helper,
    model,
    pca,
    eval_dataset_name,
    verbose=False,
):
    with PerfTimer("read_sample", PERF_STATS):
        if is_revisited_dataset(eval_dataset_name):
            img = image_helper.load_and_prepare_revisited_image(fname_in, roi=roi)
        elif is_instre_dataset(eval_dataset_name):
            img = image_helper.load_and_prepare_instre_image(fname_in)
        else:
            img = image_helper.load_and_prepare_image(fname_in, roi=roi)

    with PerfTimer("extract_features", PERF_STATS):
        # the model output is a list always.
        img_scalings = cfg.IMG_RETRIEVAL.IMG_SCALINGS or [1]
        activation_maps = extract_activation_maps(img, model, img_scalings)

    if verbose:
        print(
            f"Example eval image raw activation map shape: { activation_maps[0].shape }"  # NOQA
        )

    with PerfTimer("post_process_features", PERF_STATS):
        # process the features: rmac | l2 norm
        if cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "rmac":
            descriptors = get_rmac_descriptors(
                activation_maps[0],
                spatial_levels,
                pca=pca,
                normalize=cfg.IMG_RETRIEVAL.NORMALIZE_FEATURES,
            )
        elif cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "gem":
            descriptors = get_average_gem(
                activation_maps,
                p=cfg.IMG_RETRIEVAL.GEM_POOL_POWER,
                add_bias=True,
            )
        else:
            descriptors = torch.mean(torch.stack(activation_maps), dim=0)
            descriptors = descriptors.reshape(descriptors.shape[0], -1)

        # Optionally l2 normalize the features.
        if (
            cfg.IMG_RETRIEVAL.NORMALIZE_FEATURES
            and cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE != "rmac"
        ):
            # RMAC performs normalization within the algorithm, hence we skip it here.
            descriptors = l2n(descriptors, dim=1)

        # Optionally apply pca.
        if pca and cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE != "rmac":
            # RMAC performs pca within the algorithm, hence we skip it here.
            descriptors = pca.apply(descriptors)

    if fname_out:
        save_file(descriptors.data.numpy(), fname_out, verbose=False)
    return descriptors.data.numpy()


def get_dataset_features(
    cfg,
    temp_dir,
    eval_dataset_name,
    resize_img,
    spatial_levels,
    image_helper,
    eval_dataset,
    model,
    pca,
):
    features_dataset = []
    num_images = eval_dataset.get_num_images()
    logging.info(f"Getting features for dataset images: {num_images}")

    db_fname_out_dir = None
    if temp_dir:
        db_fname_out_dir = f"{temp_dir}/{eval_dataset_name}_S{resize_img}_db"

    makedir(db_fname_out_dir)

    for idx in range(num_images):
        if idx % LOG_FREQUENCY == 0:
            logging.info(f"Eval Dataset Image: {idx}"),
        db_fname_in = eval_dataset.get_filename(idx)

        db_fname_out = None
        if db_fname_out_dir:
            db_fname_out = f"{db_fname_out_dir}/{idx}.npy"

        if db_fname_out and g_pathmgr.exists(db_fname_out):
            db_feature = load_file(db_fname_out)
        else:
            db_feature = process_eval_image(
                cfg,
                db_fname_in,
                None,
                db_fname_out,
                spatial_levels,
                image_helper,
                model,
                pca,
                eval_dataset_name,
                verbose=(idx == 0),
            )
        features_dataset.append(db_feature)

    features_dataset = np.vstack(features_dataset)
    logging.info(f"Dataset Features Size: {features_dataset.shape}")
    return features_dataset


def get_queries_features(
    cfg,
    temp_dir,
    eval_dataset_name,
    resize_img,
    spatial_levels,
    image_helper,
    eval_dataset,
    model,
    pca,
):
    features_queries = []
    num_queries = eval_dataset.get_num_query_images()

    num_queries = (
        num_queries
        if cfg.IMG_RETRIEVAL.NUM_QUERY_SAMPLES == -1
        else cfg.IMG_RETRIEVAL.NUM_QUERY_SAMPLES
    )

    logging.info(f"Getting features for queries: {num_queries}")
    q_fname_out_dir = None
    if q_fname_out_dir:
        q_fname_out_dir = f"{temp_dir}/{eval_dataset_name}_S{resize_img}_q"
        makedir(q_fname_out_dir)

    for idx in range(num_queries):
        if idx % LOG_FREQUENCY == 0:
            logging.info(f"Eval Query: {idx}"),
        q_fname_in = eval_dataset.get_query_filename(idx)
        # Optionally crop the query by the region-of-interest (ROI).
        roi = (
            eval_dataset.get_query_roi(idx)
            if cfg.IMG_RETRIEVAL.CROP_QUERY_ROI
            else None
        )

        q_fname_out = None
        if q_fname_out_dir:
            q_fname_out = f"{q_fname_out_dir}/{idx}.npy"

        if q_fname_out and g_pathmgr.exists(q_fname_out):
            query_feature = load_file(q_fname_out)
        else:
            query_feature = process_eval_image(
                cfg,
                q_fname_in,
                roi,
                q_fname_out,
                spatial_levels,
                image_helper,
                model,
                pca,
                eval_dataset_name,
                verbose=(idx == 0),
            )
        features_queries.append(query_feature)

    features_queries = np.vstack(features_queries)
    logging.info(f"Queries Features Size: {features_queries.shape}")
    return features_queries


def get_transforms(cfg, dataset_name):
    # Setup the data transforms (basic) that we apply on the train/eval dataset.
    if is_instre_dataset(dataset_name) or is_whiten_dataset(dataset_name):
        transforms = torchvision.transforms.Compose(
            [
                MultigrainResize(int((256 / 224) * cfg.IMG_RETRIEVAL.RESIZE_IMG)),
                torchvision.transforms.CenterCrop(cfg.IMG_RETRIEVAL.RESIZE_IMG),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]

        if cfg.IMG_RETRIEVAL.CENTER_CROP:
            transforms = [
                torchvision.transforms.Resize(
                    int((256 / 224) * cfg.IMG_RETRIEVAL.RESIZE_IMG)
                ),
                torchvision.transforms.CenterCrop(cfg.IMG_RETRIEVAL.RESIZE_IMG),
            ] + transforms

        transforms = torchvision.transforms.Compose(transforms)

    return transforms


def get_train_dataset(cfg, root_dataset_path, train_dataset_name, eval_binary_path):
    # We only create the train dataset if we need PCA or whitening training.
    # Otherwise not.
    if cfg.IMG_RETRIEVAL.TRAIN_PCA_WHITENING:
        train_data_path = f"{root_dataset_path}/{train_dataset_name}"

        assert g_pathmgr.exists(train_data_path), f"Unknown path: {train_data_path}"

        num_samples = (
            None
            if cfg.IMG_RETRIEVAL.NUM_TRAINING_SAMPLES == -1
            else cfg.IMG_RETRIEVAL.NUM_TRAINING_SAMPLES
        )

        if is_revisited_dataset(train_dataset_name):
            train_dataset = RevisitedInstanceRetrievalDataset(
                train_dataset_name, root_dataset_path, num_samples=num_samples
            )
        elif is_whiten_dataset(train_dataset_name):
            train_dataset = WhiteningTrainingImageDataset(
                train_data_path,
                cfg.IMG_RETRIEVAL.WHITEN_IMG_LIST,
                num_samples=num_samples,
            )
        elif is_copdays_dataset(train_dataset_name):
            train_dataset = CopyDaysDataset(
                data_path=train_data_path,
                num_samples=num_samples,
                use_distractors=cfg.IMG_RETRIEVAL.USE_DISTRACTORS,
            )
        elif is_oxford_paris_dataset(train_dataset_name):
            train_dataset = InstanceRetrievalDataset(
                train_data_path, eval_binary_path, num_samples=num_samples
            )
        else:
            train_dataset = GenericInstanceRetrievalDataset(
                train_data_path, num_samples=num_samples
            )
    else:
        train_dataset = None
    return train_dataset


def compute_l2_distance_matrix(features_queries, features_dataset):
    """
    Computes the l2 distance of every query to every database image.
    """
    sx = np.sum(features_queries ** 2, axis=1, keepdims=True)
    sy = np.sum(features_dataset ** 2, axis=1, keepdims=True)

    return np.sqrt(-2 * features_queries.dot(features_dataset.T) + sx + sy.T)


def get_eval_dataset(cfg, root_dataset_path, eval_dataset_name, eval_binary_path):
    eval_data_path = f"{root_dataset_path}/{eval_dataset_name}"
    assert g_pathmgr.exists(eval_data_path), f"Unknown path: {eval_data_path}"

    num_samples = (
        None
        if cfg.IMG_RETRIEVAL.NUM_DATABASE_SAMPLES == -1
        else cfg.IMG_RETRIEVAL.NUM_DATABASE_SAMPLES
    )

    if is_revisited_dataset(eval_dataset_name):
        eval_dataset = RevisitedInstanceRetrievalDataset(
            eval_dataset_name, root_dataset_path, num_samples=num_samples
        )
    elif is_instre_dataset(eval_dataset_name):
        eval_dataset = InstreDataset(eval_data_path, num_samples=num_samples)
    elif is_copdays_dataset(eval_dataset_name):
        eval_dataset = CopyDaysDataset(
            data_path=eval_data_path,
            num_samples=num_samples,
            use_distractors=cfg.IMG_RETRIEVAL.USE_DISTRACTORS,
        )
    else:
        eval_dataset = InstanceRetrievalDataset(
            eval_data_path, eval_binary_path, num_samples=num_samples
        )
    return eval_dataset


def instance_retrieval_test(args, cfg):
    # We require 1-gpu for feature extraction. Hence check CUDA is available.
    assert torch.cuda.is_available(), "CUDA not available, Exit!"

    train_dataset_name = cfg.IMG_RETRIEVAL.TRAIN_DATASET_NAME
    eval_dataset_name = cfg.IMG_RETRIEVAL.EVAL_DATASET_NAME
    spatial_levels = cfg.IMG_RETRIEVAL.SPATIAL_LEVELS
    resize_img = cfg.IMG_RETRIEVAL.RESIZE_IMG
    eval_binary_path = cfg.IMG_RETRIEVAL.EVAL_BINARY_PATH
    root_dataset_path = cfg.IMG_RETRIEVAL.DATASET_PATH
    save_features = cfg.IMG_RETRIEVAL.SAVE_FEATURES

    temp_dir = None
    if save_features:
        temp_dir = os.path.join(get_checkpoint_folder(cfg), "features")
        logging.info(f"Temp directory: {temp_dir}")

    ############################################################################
    # Step 1: Prepare the train/eval datasets, create model and load weights
    # We only create the train dataset if we need PCA/whitening otherwise
    # train_dataset is None
    train_dataset = get_train_dataset(
        cfg, root_dataset_path, train_dataset_name, eval_binary_path
    )

    # create the eval dataset. INSTRE data evaluation requires whitening.
    eval_dataset = get_eval_dataset(
        cfg, root_dataset_path, eval_dataset_name, eval_binary_path
    )

    # Setup the data transforms (basic) that we apply on the train/eval dataset.
    transforms = get_transforms(cfg, eval_dataset_name)

    # Create the image helper
    image_helper = InstanceRetrievalImageLoader(
        S=resize_img, transforms=transforms, center_crop=cfg.IMG_RETRIEVAL.CENTER_CROP
    )

    # Build the model on gpu and set in the eval mode
    model = build_retrieval_model(cfg)
    model = copy_model_to_gpu(model)

    logging.info("Freezing the model.....")
    model.eval()
    model.freeze_head_and_trunk()

    ############################################################################
    # Step 2: Extract the features for the train dataset, calculate PCA or
    # whitening and save
    if cfg.IMG_RETRIEVAL.TRAIN_PCA_WHITENING:
        logging.info("Extracting training features...")
        # the features are already processed based on type: rmac | GeM | l2 norm
        with PerfTimer("get_train_features", PERF_STATS):
            train_features = get_train_features(
                cfg,
                temp_dir,
                train_dataset_name,
                resize_img,
                spatial_levels,
                image_helper,
                train_dataset,
                model,
            )
        ########################################################################
        # Train PCA on the train features
        pca_out_fname = None
        if temp_dir:

            pca_out_fname = f"{temp_dir}/{train_dataset_name}_S{resize_img}_PCA.pickle"
        if pca_out_fname and g_pathmgr.exists(pca_out_fname):
            logging.info("Loading PCA...")
            pca = load_pca(pca_out_fname)
        else:
            logging.info("Training and saving PCA...")
            pca = train_and_save_pca(
                train_features, cfg.IMG_RETRIEVAL.N_PCA, pca_out_fname
            )
    else:
        pca = None

    ############################################################################
    # Step 4: Extract db_features and q_features for the eval dataset
    with PerfTimer("get_query_features", PERF_STATS):
        logging.info("Extracting Queries features...")
        features_queries = get_queries_features(
            cfg,
            temp_dir,
            eval_dataset_name,
            resize_img,
            spatial_levels,
            image_helper,
            eval_dataset,
            model,
            pca,
        )

    with PerfTimer("get_dataset_features", PERF_STATS):
        logging.info("Extracting Dataset features...")
        features_dataset = get_dataset_features(
            cfg,
            temp_dir,
            eval_dataset_name,
            resize_img,
            spatial_levels,
            image_helper,
            eval_dataset,
            model,
            pca,
        )

    ############################################################################
    # Step 5: Compute similarity, score, and save results
    with PerfTimer("scoring_results", PERF_STATS):
        logging.info("Calculating similarity and score...")

        if cfg.IMG_RETRIEVAL.SIMILARITY_MEASURE == "cosine_similarity":
            sim = features_queries.dot(features_dataset.T)
        elif cfg.IMG_RETRIEVAL.SIMILARITY_MEASURE == "l2":
            sim = -compute_l2_distance_matrix(features_queries, features_dataset)
        else:
            raise ValueError(f"{ cfg.IMG_RETRIEVAL.SIMILARITY_MEASURE } not supported.")

        logging.info(f"Similarity tensor: {sim.shape}")
        results = eval_dataset.score(sim, temp_dir)

    ############################################################################
    # Step 6: save results and cleanup the temp directory
    if cfg.IMG_RETRIEVAL.SAVE_RETRIEVAL_RANKINGS_SCORES:
        checkpoint_folder = get_checkpoint_folder(cfg)

        # Save the rankings
        sim = sim.T
        ranks = np.argsort(-sim, axis=0)
        save_file(ranks.T.tolist(), os.path.join(checkpoint_folder, "rankings.json"))

        # Save the similarity scores
        save_file(
            sim.tolist(), os.path.join(checkpoint_folder, "similarity_scores.json")
        )
        # Save the result metrics
        save_file(
            results,
            os.path.join(checkpoint_folder, "metrics.json"),
            append_to_json=False,
        )

    logging.info("All done!!")


def validate_and_infer_config(config: AttrDict):
    if config.IMG_RETRIEVAL.DEBUG_MODE:
        # Set data limits for the number of training, query, and database samples.
        if config.IMG_RETRIEVAL.NUM_TRAINING_SAMPLES == -1:
            config.IMG_RETRIEVAL.NUM_TRAINING_SAMPLES = 10

        if config.IMG_RETRIEVAL.NUM_QUERY_SAMPLES == -1:
            config.IMG_RETRIEVAL.NUM_QUERY_SAMPLES = 10

        if config.IMG_RETRIEVAL.NUM_DATABASE_SAMPLES == -1:
            config.IMG_RETRIEVAL.NUM_DATABASE_SAMPLES = 50

    if config.IMG_RETRIEVAL.EVAL_DATASET_NAME in ["OXFORD", "PARIS"]:
        # InstanceRetrievalDataset#score requires the features to be saved.
        config.IMG_RETRIEVAL.SAVE_FEATURES = True

    assert (
        config.IMG_RETRIEVAL.SPATIAL_LEVELS > 0
    ), "Spatial levels must be greater than 0."
    if config.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "rmac":
        assert (
            config.IMG_RETRIEVAL.TRAIN_PCA_WHITENING
        ), "PCA Whitening is built-in to the RMAC algorithm and is required"
        assert (
            len(config.IMG_RETRIEVAL.IMG_SCALINGS) == 1
        ), "Multiple image scalings is not compatible with the rmac algorithm."

    assert config.IMG_RETRIEVAL.SIMILARITY_MEASURE in ["cosine_similarity", "l2"]

    return config


def main(args: Namespace, config: AttrDict):
    config = validate_and_infer_config(config)
    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=config)

    # setup the logging
    checkpoint_folder = get_checkpoint_folder(config)
    setup_logging(__name__, output_dir=checkpoint_folder)

    # print the config
    print_cfg(config)

    instance_retrieval_test(args, config)
    logging.info(f"Performance time breakdow:\n{PERF_STATS.report_str()}")

    # close the logging streams including the filehandlers
    shutdown_logging()


def hydra_main(overrides: List[Any]):
    cfg = compose_hydra_configuration(overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


if __name__ == "__main__":
    overrides = sys.argv[1:]
    hydra_main(overrides=overrides)
