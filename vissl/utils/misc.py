# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import random
import tempfile

import numpy as np
import pkg_resources
import torch
import torch.multiprocessing as mp
from scipy.sparse import csr_matrix
from vissl.utils.io import load_file


def is_apex_available():
    """
    Check if apex is available with simple python imports.
    """
    try:
        import apex  # NOQA

        apex_available = True
    except ImportError:
        apex_available = False
    return apex_available


def find_free_tcp_port():
    """
    Find the free port that can be used for Rendezvous on the local machine.
    We use this for 1 machine training where the port is automatically detected.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def get_dist_run_id(cfg, num_nodes):
    """
    For multi-gpu training with PyTorch, we have to specify
    how the gpus are going to rendezvous. This requires specifying
    the communication method: file, tcp and the unique rendezvous run_id that
    is specific to 1 run.

    We recommend:
        1) for 1-node: use init_method=tcp and run_id=auto
        2) for multi-node, use init_method=tcp and specify run_id={master_node}:{port}
    """
    init_method = cfg.DISTRIBUTED.INIT_METHOD
    run_id = cfg.DISTRIBUTED.RUN_ID
    if init_method == "tcp" and cfg.DISTRIBUTED.RUN_ID == "auto":
        assert (
            num_nodes == 1
        ), "cfg.DISTRIBUTED.RUN_ID=auto is allowed for 1 machine only."
        port = find_free_tcp_port()
        run_id = f"localhost:{port}"
    elif init_method == "file":
        if num_nodes > 1:
            logging.warning(
                "file is not recommended to use for distributed training on > 1 node"
            )
        # Find a unique tempfile if needed.
        if not run_id or run_id == "auto":
            unused_fno, run_id = tempfile.mkstemp()
    elif init_method == "tcp" and cfg.DISTRIBUTED.NUM_NODES > 1:
        assert cfg.DISTRIBUTED.RUN_ID, "please specify RUN_ID for tcp"
    elif init_method == "env":
        assert num_nodes == 1, "can not use 'env' init method for multi-node. Use tcp"
    return run_id


def setup_multiprocessing_method(method_name: str):
    """
    PyTorch supports several multiprocessing options: forkserver | spawn | fork

    We recommend and use forkserver as the default method in VISSL.
    """
    try:
        mp.set_start_method(method_name, force=True)
        logging.info("Set start method of multiprocessing to {}".format(method_name))
    except RuntimeError:
        pass


def set_seeds(cfg, node_id=0):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    node_seed = cfg.SEED_VALUE
    if cfg.DISTRIBUTED.NUM_NODES > 1:
        node_seed = node_seed * 2 * node_id
    logging.info(f"MACHINE SEED: {node_seed}")
    random.seed(node_seed)
    np.random.seed(node_seed)
    torch.manual_seed(node_seed)
    if cfg["MACHINE"]["DEVICE"] == "gpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(node_seed)


def get_indices_sparse(data):
    """
    Is faster than np.argwhere. Used in loss functions like swav loss, etc
    """
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


def merge_features(output_dir, split, layer, cfg):
    """
    For multi-gpu feature extraction, each gpu saves features corresponding to its
    share of the data. We can merge the features across all gpus to get the features
    for the full data.

    The features are saved along with the data indexes and label. The data indexes can
    be used to sort the data and ensure the uniqueness.

    We organize the features, targets corresponding to the data index of each feature,
    ensure the uniqueness and return.

    Args:
        output_dir (str): input path where the features are dumped
        split (str): whether the features are train or test data features
        layer (str): the features correspond to what layer of the model
        cfg (AttrDict): the input configuration specified by user

    Returns:
        output (Dict): contains features, targets, inds as the keys
    """
    logging.info(f"Merging features: {split} {layer}")
    output_feats, output_targets = {}, {}
    for local_rank in range(0, cfg.DISTRIBUTED.NUM_PROC_PER_NODE):
        for node_id in range(0, cfg.DISTRIBUTED.NUM_NODES):
            dist_rank = cfg.DISTRIBUTED.NUM_PROC_PER_NODE * node_id + local_rank
            feat_file = f"{output_dir}/rank{dist_rank}_{split}_{layer}_features.npy"
            targets_file = f"{output_dir}/rank{dist_rank}_{split}_{layer}_targets.npy"
            inds_file = f"{output_dir}/rank{dist_rank}_{split}_{layer}_inds.npy"
            logging.info(f"Loading:\n{feat_file}\n{targets_file}\n{inds_file}")
            feats = load_file(feat_file)
            targets = load_file(targets_file)
            indices = load_file(inds_file)
            num_samples = feats.shape[0]
            for idx in range(num_samples):
                index = indices[idx]
                if not (index in output_feats):
                    output_feats[index] = feats[idx]
                    output_targets[index] = targets[idx]
    output = {}
    output_feats = dict(sorted(output_feats.items()))
    output_targets = dict(sorted(output_targets.items()))
    feats = np.array(list(output_feats.values()))
    N = feats.shape[0]
    output = {
        "features": feats.reshape(N, -1),
        "targets": np.array(list(output_targets.values())),
        "inds": np.array(list(output_feats.keys())),
    }
    logging.info(f"Features: {output['features'].shape}")
    logging.info(f"Targets: {output['targets'].shape}")
    logging.info(f"Indices: {output['inds'].shape}")
    return output


def get_json_data_catalog_file():
    """
    Searches for the dataset_catalog.json file that contains information about
    the dataset paths if set by user.
    """
    json_catalog_path = pkg_resources.resource_filename(
        "configs", "config/dataset_catalog.json"
    )
    return json_catalog_path


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
