# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections.abc
import logging
import os
import random
import sys
import tempfile
import time
from functools import partial, wraps
from typing import Tuple

import numpy as np
import pkg_resources
import torch
import torch.multiprocessing as mp
from iopath.common.file_io import g_pathmgr
from scipy.sparse import csr_matrix
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader


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


def is_augly_available():
    """
    Check if apex is available with simple python imports.
    """
    try:
        assert sys.version_info >= (
            3,
            7,
            0,
        ), "Please upgrade your python version to 3.7 or higher to use Augly."

        import augly.image  # NOQA

        augly_available = True
    except (AssertionError, ImportError):
        augly_available = False
    return augly_available


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
    torch_seed = torch.utils.data.get_worker_info().seed % (2**32)
    random.seed(torch_seed)
    np.random.seed(torch_seed)


def get_indices_sparse(data):
    """
    Is faster than np.argwhere. Used in loss functions like swav loss, etc
    """
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


def merge_features(input_dir: str, split: str, layer: str):
    return ExtractedFeaturesLoader.load_features(input_dir, split, layer)


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
        assert g_pathmgr.exists(
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


class set_torch_seed:
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


# Credit: https://stackoverflow.com/questions/42521549/retry-function-in-python
def retry(func=None, exception=Exception, n_tries=5, delay=5, backoff=1, logger=False):
    """Retry decorator with exponential backoff.

    Parameters
    ----------
    func : typing.Callable, optional
        Callable on which the decorator is applied, by default None
    exception : Exception or tuple of Exceptions, optional
        Exception(s) that invoke retry, by default Exception
    n_tries : int, optional
        Number of tries before giving up, by default 5
    delay : int, optional
        Initial delay between retries in seconds, by default 5
    backoff : int, optional
        Backoff multiplier e.g. value of 2 will double the delay, by default 1
    logger : bool, optional
        Option to log or print, by default False

    Returns
    -------
    typing.Callable
        Decorated callable that calls itself when exception(s) occur.

    Examples
    --------
    >>> import random
    >>> @retry(exception=Exception, n_tries=4)
    ... def test_random(text):
    ...    x = random.random()
    ...    if x < 0.5:
    ...        raise Exception("Fail")
    ...    else:
    ...        print("Success: ", text)
    >>> test_random("It works!")
    """

    if func is None:
        return partial(
            retry,
            exception=exception,
            n_tries=n_tries,
            delay=delay,
            backoff=backoff,
            logger=logger,
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        ntries, ndelay = n_tries, delay

        while ntries > 1:
            try:
                return func(*args, **kwargs)
            except exception as e:
                msg = f"{str(e)}, Retrying in {ndelay} seconds..."
                if logger:
                    logging.warning(msg)
                else:
                    print(msg)
                time.sleep(ndelay)
                ntries -= 1
                ndelay *= backoff

        return func(*args, **kwargs)

    return wrapper


# Credit: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys # NOQA
def flatten_dict(d: dict, parent_key="", sep="_"):
    """
    Flattens a dict, delimited with a '_'. For example the input:
    {
        'top_1': {
            'res_5': 100
        }
    }

    will return:

    {
        'top_1_res_5': 100
    }
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Credit: https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
def recursive_dict_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            dict1[k] = recursive_dict_merge(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1


def torch_version() -> Tuple[int, ...]:
    numbering = torch.__version__.split("+")[0].split(".")[:3]

    # Catch torch version if run against internal pre-releases, like `1.8.0a0fb`,
    if not numbering[2].isnumeric():
        # Two options here:
        # - either skip this version (minor number check is not relevant)
        # - or check that our codebase is not broken by this ongoing development.

        # Assuming that we're interested in the second usecase more than the first,
        # return the pre-release or dev numbering
        numbering[2] = "0"

    return tuple(int(n) for n in numbering)
