# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import random
import tempfile

import numpy as np
import pkg_resources
import torch
import torch.multiprocessing as mp
from fvcore.common.file_io import PathManager
from scipy.sparse import csr_matrix
from vissl.utils.io import load_file


def is_fairscale_sharded_available():
    """
    Check if the fairscale version has the ShardedGradScaler()
    to use with ZeRO + PyTorchAMP
    """
    try:
        from fairscale.optim.grad_scaler import ShardedGradScaler  # NOQA

        fairscale_sharded_available = True
    except ImportError:
        fairscale_sharded_available = False
    return fairscale_sharded_available


def is_faiss_available():
    """
    Check if faiss is available with simple python imports.

    To install faiss, simply do:
        If using PIP env: `pip install faiss-gpu`
        If using conda env: `conda install faiss-gpu -c pytorch`
    """
    try:
        import faiss  # NOQA

        faiss_available = True
    except ImportError:
        faiss_available = False
    return faiss_available


def is_opencv_available():
    """
    Check if opencv is available with simple python imports.

    To install opencv, simply do: `pip install opencv-python`
    regardless of whether using conda or pip environment.
    """
    try:
        import cv2  # NOQA

        opencv_available = True
    except ImportError:
        opencv_available = False
    return opencv_available


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


def set_seeds(cfg, dist_rank):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    # Since in the pytorch sampler, we increment the seed by 1 for every epoch.
    seed_value = (cfg.SEED_VALUE + dist_rank) * cfg.OPTIMIZER.num_epochs
    logging.info(f"MACHINE SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if cfg["MACHINE"]["DEVICE"] == "gpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def set_dataloader_seeds(_worker_id: int):
    """
    See: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    When using "Fork" process spawning, the dataloader workers inherit the seeds of the
    parent process for numpy. While torch seeds are handled correctly across dataloaders and
    across epochs, numpy seeds are not. Therefore in order to ensure each worker has a
    different and deterministic seed, we must explicitly set the numpy seed to the torch seed.
    Also see https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading
    """
    # numpy and random seed must be between 0 and 2 ** 32 - 1.
    torch_seed = torch.utils.data.get_worker_info().seed % (2 ** 32)
    random.seed(torch_seed)
    np.random.seed(torch_seed)


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


def get_json_catalog_path(default_dataset_catalog_path: str) -> str:
    """
    Gets dataset catalog json file absolute path.
    Optionally set environment variable VISSL_DATASET_CATALOG_PATH for dataset catalog path.
    Useful for local development and/or remote server configuration.
    """
    dataset_catalog_path = os.environ.get(
        "VISSL_DATASET_CATALOG_PATH", default_dataset_catalog_path
    )

    # If catalog path is the default and we cannot find it, we want to continue without failing.
    if os.environ.get("VISSL_DATASET_CATALOG_PATH", False):
        assert PathManager.exists(
            dataset_catalog_path
        ), f"Dataset catalog path: { dataset_catalog_path } not found."

    return dataset_catalog_path


def get_json_data_catalog_file():
    """
    Searches for the dataset_catalog.json file that contains information about
    the dataset paths if set by user.
    """
    default_path = pkg_resources.resource_filename(
        "configs", "config/dataset_catalog.json"
    )
    json_catalog_path = get_json_catalog_path(default_path)

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


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)
