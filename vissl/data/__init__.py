# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random

import numpy as np
import torch
from classy_vision.dataset import DataloaderAsyncGPUWrapper
from torch.utils.data import DataLoader
from vissl.data.airstore_dataset import AirstoreDataset
from vissl.data.collators import get_collator
from vissl.data.data_helper import (
    DeterministicDistributedSampler,
    StatefulDistributedSampler,
)
from vissl.data.dataloader_sync_gpu_wrapper import DataloaderSyncGPUWrapper
from vissl.data.dataset_catalog import (
    get_data_files,
    register_datasets,
    VisslDatasetCatalog,
)
from vissl.data.disk_dataset import DiskImageDataset
from vissl.data.ssl_dataset import GenericSSLDataset
from vissl.data.synthetic_dataset import SyntheticImageDataset
from vissl.data.torchvision_dataset import TorchvisionDataset
from vissl.utils.misc import set_dataloader_seeds, setup_multiprocessing_method


__all__ = [
    "AirstoreDataset",
    "GenericSSLDataset",
    "get_data_files",
    "register_datasets",
    "VisslDatasetCatalog",
]

DATASET_SOURCE_MAP = {
    "airstore": AirstoreDataset,
    "disk_filelist": DiskImageDataset,
    "disk_folder": DiskImageDataset,
    "disk_roi_annotations": DiskImageDataset,
    "torchvision_dataset": TorchvisionDataset,
    "synthetic": SyntheticImageDataset,
}


DATA_SOURCES_WITH_SUBSET_SUPPORT = {
    "disk_filelist",
    "disk_folder",
    "disk_roi_annotations",
    "torchvision_dataset",
    "synthetic",
}


def build_dataset(cfg, split, **kwargs):
    """
    Given the user defined config and the dataset split (train/val), build
    the dataset.

    Args:
        cfg (AttrDict): user defined configuration file
        split (str): dataset split (from train or test)

    Returns:
        Instance of GenericSSLDataset
    """
    return GenericSSLDataset(
        cfg=cfg,
        split=split,
        dataset_source_map=DATASET_SOURCE_MAP,
        data_sources_with_subset=DATA_SOURCES_WITH_SUBSET_SUPPORT,
    )


def print_sampler_config(data_sampler):
    """
    Print the data sampler config to facilitate debugging. Printed params include:
        num_replicas,
        rank
        epoch
        num_samples
        total_size
        shuffle
    """
    sampler_cfg = {
        "num_replicas": data_sampler.num_replicas,
        "rank": data_sampler.rank,
        "epoch": data_sampler.epoch,
        "num_samples": data_sampler.num_samples,
        "total_size": data_sampler.total_size,
        "shuffle": data_sampler.shuffle,
    }
    if hasattr(data_sampler, "start_iter"):
        sampler_cfg["start_iter"] = data_sampler.start_iter
    if hasattr(data_sampler, "batch_size"):
        sampler_cfg["batch_size"] = data_sampler.batch_size
    if hasattr(data_sampler, "seed"):
        sampler_cfg["seed"] = data_sampler.seed
    logging.info("Distributed Sampler config:\n{}".format(sampler_cfg))


def get_sampler(dataset, dataset_config, sampler_seed=0):
    """
    Given the dataset object and the dataset config, get the data sampler to use
    Supports 2 types of samplers:
        - Pytorch default torch.utils.data.distributed.DistributedSampler
        - VISSL sampler StatefulDistributedSampler that is written specifically for
          large scale dataset trainings
    """
    data_sampler = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if dataset_config["USE_DEBUGGING_SAMPLER"]:
            data_sampler = DeterministicDistributedSampler(
                dataset, batch_size=dataset_config["BATCHSIZE_PER_REPLICA"]
            )
        elif dataset_config["USE_STATEFUL_DISTRIBUTED_SAMPLER"]:
            data_sampler = StatefulDistributedSampler(
                dataset,
                batch_size=dataset_config["BATCHSIZE_PER_REPLICA"],
                seed=sampler_seed,
            )
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        logging.info("Created the Distributed Sampler....")
        print_sampler_config(data_sampler)
    else:
        logging.warning(
            "Distributed trainer not initialized. Not using the sampler and data will NOT be shuffled"  # NOQA
        )
    return data_sampler


def debugging_worker_init_fn(worker_id: int):
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)


def build_dataloader(
    dataset: GenericSSLDataset,
    dataset_config: dict,
    num_dataloader_workers: int,
    pin_memory: bool,
    multi_processing_method: str,
    device: torch.device,
    sampler_seed=0,
    get_sampler=get_sampler,
    worker_init_fn=set_dataloader_seeds,
    **kwargs,
):
    """
    Get the dataloader for the given satasets and data split

    Args:
        dataset (GenericSSLDataset):    the dataset object for which dataloader is constructed
        dataset_config (dict):          configuration of the dataset.
                                        should be DATA.TRAIN or DATA.TEST settings
        num_dataloader_workers (int):   number of workers per gpu (or cpu) training
        pin_memory (bool):              whether to pin memory or not
        multi_processing_method (str):  method to use. options: forkserver | fork | spawn
        sampler_seed (int):             seed for the sampler. Should be identical per process
        device (torch.device):          training on cuda or cpu
        get_sampler (get_sampler):      function that is used to get the sampler
        worker_init_fn (None):          any function that should be executed during
                                        initialization of dataloader workers

    Returns:
        Instance of Pytorch DataLoader. The dataloader is wrapped with
        DataloaderAsyncGPUWrapper or DataloaderSyncGPUWrapper depending
        on whether user wants to copy data to gpu async or not.
    """

    # pytorch dataloader requires setting the multiprocessing type.
    setup_multiprocessing_method(multi_processing_method)

    # we don't need to set the rank, replicas as the Sampler already does so in
    # it's init function
    data_sampler = get_sampler(dataset, dataset_config, sampler_seed)
    collate_function = get_collator(
        dataset_config["COLLATE_FUNCTION"], dataset_config["COLLATE_FUNCTION_PARAMS"]
    )

    # Replace the worker_init_fn with a deterministic one when debugging
    if dataset_config["USE_DEBUGGING_SAMPLER"]:
        worker_init_fn = debugging_worker_init_fn

    # Load the labels of the dataset before creating the data loader
    # or else the load of files will happen on each data loader separately
    # decreasing performance / hitting quota on data source
    dataset.load_labels()

    # Create the pytorch dataloader
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=num_dataloader_workers,
        pin_memory=pin_memory,
        shuffle=False,
        batch_size=dataset_config["BATCHSIZE_PER_REPLICA"],
        collate_fn=collate_function,
        sampler=data_sampler,
        drop_last=dataset_config["DROP_LAST"],
        worker_init_fn=worker_init_fn,
    )
    enable_async_gpu_copy = dataset.cfg["DATA"]["ENABLE_ASYNC_GPU_COPY"]
    dataloader = wrap_dataloader(dataloader, enable_async_gpu_copy, device)

    return dataloader


def wrap_dataloader(dataloader, enable_async_gpu_copy: bool, device: torch.device):
    """
    If the targeted device is CUDA, set up async device copy:
        - makes sure that samples are on device
        - overlap the copy with the previous batch computation.
    """
    if device.type == "cuda":
        if enable_async_gpu_copy:
            logging.info("Wrapping the dataloader to async device copies")  # NOQA
            dataloader = DataloaderAsyncGPUWrapper(dataloader)
        else:
            logging.info("Wrapping the dataloader to synchronous device copies")  # NOQA
            dataloader = DataloaderSyncGPUWrapper(dataloader)
    else:
        logging.warning("Selecting a CPU device")

    return dataloader
