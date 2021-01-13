# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import sys
import uuid
from argparse import Namespace
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classy_vision.generic.util import copy_model_to_gpu
from fvcore.common.file_io import PathManager
from hydra.experimental import compose, initialize_config_module
from vissl.models import build_model
from vissl.utils.checkpoint import init_model_from_weights
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import (
    AttrDict,
    convert_to_attrdict,
    is_hydra_available,
    print_cfg,
)
from vissl.utils.instance_retrieval_utils.data_util import (
    InstanceRetrievalDataset,
    InstanceRetrievalImageLoader,
    InstreDataset,
    MultigrainResize,
    RevisitedInstanceRetrievalDataset,
    WhiteningTrainingImageDataset,
    gem,
    is_instre_dataset,
    is_revisited_dataset,
    is_whiten_dataset,
    l2n,
)
from vissl.utils.instance_retrieval_utils.pca import load_pca, train_and_save_pca
from vissl.utils.instance_retrieval_utils.rmac import get_rmac_descriptors
from vissl.utils.io import cleanup_dir, load_file, makedir, save_file
from vissl.utils.logger import setup_logging, shutdown_logging


# frequency at which we log the image number being processed.
LOG_FREQUENCY = 100


def build_retrieval_model(cfg):
    """
    Builds the model on 1-gpu and initializes from the weight.
    """
    logging.info("Building model....")
    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    if PathManager.exists(cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE):
        init_weights_path = cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE
        logging.info(f"Initializing model from: {init_weights_path}")
        weights = torch.load(init_weights_path, map_location=torch.device("cuda"))
        skip_layers = cfg.MODEL.WEIGHTS_INIT.get("SKIP_LAYERS", [])
        replace_prefix = cfg.MODEL.WEIGHTS_INIT.get("REMOVE_PREFIX", None)
        append_prefix = cfg.MODEL.WEIGHTS_INIT.get("APPEND_PREFIX", None)
        state_dict_key_name = cfg.MODEL.WEIGHTS_INIT.get("STATE_DICT_KEY_NAME", None)

        init_model_from_weights(
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
        logging.warn("Model is randomly initialized....")
    logging.info(f"Model is:\n {model}")
    return model


def gem_pool_and_save_features(features, p, add_bias, gem_out_fname):
    if PathManager.exists(gem_out_fname):
        logging.info("Loading train GeM features...")
        features = load_file(gem_out_fname)
    else:
        logging.info(f"GeM pooling features: {features.shape}")
        features = l2n(gem(features, p=p, add_bias=True))
        save_file(features, gem_out_fname)
        logging.info(f"Saved GeM features to: {gem_out_fname}")
    return features


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

    def process_train_image(i, out_dir):
        if i % LOG_FREQUENCY == 0:
            logging.info(f"Train Image: {i}"),
        fname_out = f"{out_dir}/{i}.npy"
        if PathManager.exists(fname_out):
            feat = load_file(fname_out)
            train_features.append(feat)
        else:
            fname_in = train_dataset.get_filename(i)
            if is_revisited_dataset(train_dataset_name):
                img = image_helper.load_and_prepare_revisited_image(fname_in)
            elif is_whiten_dataset(train_dataset_name):
                img = image_helper.load_and_prepare_whitening_image(fname_in)
            else:
                img = image_helper.load_and_prepare_image(fname_in, roi=None)
            v = torch.autograd.Variable(img.unsqueeze(0))
            vc = v.cuda()
            # the model output is a list always.
            activation_map = model(vc)[0].cpu()
            # once we have the features,
            # we can perform: rmac | gem pooling | l2 norm
            if cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "rmac":
                descriptors = get_rmac_descriptors(activation_map, spatial_levels)
            else:
                descriptors = activation_map
            save_file(descriptors.data.numpy(), fname_out)
            train_features.append(descriptors.data.numpy())

    num_images = train_dataset.get_num_images()
    out_dir = f"{temp_dir}/{train_dataset_name}_S{resize_img}_features_train"
    makedir(out_dir)
    for i in range(num_images):
        process_train_image(i, out_dir)

    if cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "gem":
        gem_out_fname = f"{out_dir}/{train_dataset_name}_GeM.npy"
        train_features = torch.tensor(np.concatenate(train_features))
        train_features = gem_pool_and_save_features(
            train_features,
            p=cfg.IMG_RETRIEVAL.GEM_POOL_POWER,
            add_bias=True,
            gem_out_fname=gem_out_fname,
        )
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
):
    if is_revisited_dataset(eval_dataset_name):
        img = image_helper.load_and_prepare_revisited_image(fname_in, roi=roi)
    elif is_instre_dataset(eval_dataset_name):
        img = image_helper.load_and_prepare_instre_image(fname_in)
    else:
        img = image_helper.load_and_prepare_image(fname_in, roi=roi)
    v = torch.autograd.Variable(img.unsqueeze(0))
    vc = v.cuda()
    # the model output is a list always.
    activation_map = model(vc)[0].cpu()
    # process the features: rmac | l2 norm
    if cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "rmac":
        descriptors = get_rmac_descriptors(activation_map, spatial_levels, pca=pca)
    elif cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "l2_norm":
        # we simply L2 normalize the features otherwise
        descriptors = F.normalize(activation_map, p=2, dim=0)
    else:
        descriptors = activation_map
    save_file(descriptors.data.numpy(), fname_out)
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
    db_fname_out_dir = "{}/{}_S{}_db".format(temp_dir, eval_dataset_name, resize_img)
    makedir(db_fname_out_dir)

    for idx in range(num_images):
        if idx % LOG_FREQUENCY == 0:
            logging.info(f"Eval Dataset Image: {idx}"),
        db_fname_in = eval_dataset.get_filename(idx)
        db_fname_out = f"{db_fname_out_dir}/{idx}.npy"
        if PathManager.exists(db_fname_out):
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
            )
        features_dataset.append(db_feature)

    if cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "gem":
        # GeM pool the features and apply the PCA
        gem_out_fname = f"{db_fname_out_dir}/{eval_dataset_name}_GeM.npy"
        features_dataset = torch.tensor(np.concatenate(features_dataset))
        features_dataset = gem_pool_and_save_features(
            features_dataset,
            p=cfg.IMG_RETRIEVAL.GEM_POOL_POWER,
            add_bias=True,
            gem_out_fname=gem_out_fname,
        )
        features_dataset = pca.apply(features_dataset)
    features_dataset = np.vstack(features_dataset)
    logging.info(f"features dataset: {features_dataset.shape}")
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
    if cfg.IMG_RETRIEVAL.DEBUG_MODE:
        num_queries = 50
    logging.info(f"Getting features for queries: {num_queries}")
    q_fname_out_dir = "{}/{}_S{}_q".format(temp_dir, eval_dataset_name, resize_img)
    makedir(q_fname_out_dir)

    for idx in range(num_queries):
        if idx % LOG_FREQUENCY == 0:
            logging.info(f"Eval Query: {idx}"),
        q_fname_in = eval_dataset.get_query_filename(idx)
        roi = eval_dataset.get_query_roi(idx)
        q_fname_out = f"{q_fname_out_dir}/{idx}.npy"
        if PathManager.exists(q_fname_out):
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
            )
        features_queries.append(query_feature)

    if cfg.IMG_RETRIEVAL.FEATS_PROCESSING_TYPE == "gem":
        # GeM pool the features and apply the PCA
        gem_out_fname = f"{q_fname_out_dir}/{eval_dataset_name}_GeM.npy"
        features_queries = torch.tensor(np.concatenate(features_queries))
        features_queries = gem_pool_and_save_features(
            features_queries,
            p=cfg.IMG_RETRIEVAL.GEM_POOL_POWER,
            add_bias=True,
            gem_out_fname=gem_out_fname,
        )
        features_queries = pca.apply(features_queries)
    features_queries = np.vstack(features_queries)
    logging.info(f"features queries: {features_queries.shape}")
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
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms


def get_train_dataset(cfg, root_dataset_path, train_dataset_name, eval_binary_path):
    # We only create the train dataset if we need PCA or whitening training.
    # Otherwise not.
    if cfg.IMG_RETRIEVAL.SHOULD_TRAIN_PCA_OR_WHITENING:
        train_data_path = f"{root_dataset_path}/{train_dataset_name}"
        assert PathManager.exists(train_data_path), f"Unknown path: {train_data_path}"

        num_samples = 10 if cfg.IMG_RETRIEVAL.DEBUG_MODE else None

        if is_revisited_dataset(train_dataset_name):
            train_dataset = RevisitedInstanceRetrievalDataset(
                train_dataset_name, root_dataset_path
            )
        elif is_whiten_dataset(train_dataset_name):
            train_dataset = WhiteningTrainingImageDataset(
                train_data_path,
                cfg.IMG_RETRIEVAL.WHITEN_IMG_LIST,
                num_samples=num_samples,
            )
        else:
            train_dataset = InstanceRetrievalDataset(
                train_data_path, eval_binary_path, num_samples=num_samples
            )
    else:
        train_dataset = None
    return train_dataset


def get_eval_dataset(cfg, root_dataset_path, eval_dataset_name, eval_binary_path):
    eval_data_path = f"{root_dataset_path}/{eval_dataset_name}"
    assert PathManager.exists(eval_data_path), f"Unknown path: {eval_data_path}"

    num_samples = 20 if cfg.IMG_RETRIEVAL.DEBUG_MODE else None

    if is_revisited_dataset(eval_dataset_name):
        eval_dataset = RevisitedInstanceRetrievalDataset(
            eval_dataset_name, root_dataset_path
        )
    elif is_instre_dataset(eval_dataset_name):
        eval_dataset = InstreDataset(eval_data_path, num_samples=num_samples)
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
    temp_dir = f"{cfg.IMG_RETRIEVAL.TEMP_DIR}/{str(uuid.uuid4())}"
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
    image_helper = InstanceRetrievalImageLoader(S=resize_img, transforms=transforms)

    # Build the model on gpu and set in the eval mode
    model = build_retrieval_model(cfg)
    model = copy_model_to_gpu(model)

    logging.info("Freezing the model.....")
    model.eval()
    model.freeze_head_and_trunk()

    ############################################################################
    # Step 2: Extract the features for the train dataset, calculate PCA or
    # whitening and save
    if cfg.IMG_RETRIEVAL.SHOULD_TRAIN_PCA_OR_WHITENING:
        logging.info("Extracting training features...")
        # the features are already processed based on type: rmac | GeM | l2 norm
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
        pca_out_fname = f"{temp_dir}/{train_dataset_name}_S{resize_img}_PCA.pickle"
        if PathManager.exists(pca_out_fname):
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
    # Step 5: Compute similarity and score
    logging.info("Calculating similarity and score...")
    sim = features_queries.dot(features_dataset.T)
    logging.info(f"Similarity tensor: {sim.shape}")
    eval_dataset.score(sim, temp_dir)

    ############################################################################
    # Step 6: cleanup the temp directory
    logging.info(f"Cleaning up temp directory: {temp_dir}")
    cleanup_dir(temp_dir)

    logging.info("All done!!")


def main(args: Namespace, config: AttrDict):
    # setup the logging
    setup_logging(__name__)

    # print the config
    print_cfg(config)

    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=config)

    instance_retrieval_test(args, config)
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
