# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import torch
from classy_vision.dataset import DataloaderAsyncGPUWrapper
from torch.utils.data import DataLoader
from vissl.data.collators import get_collator
from vissl.data.data_helper import StatefulDistributedSampler
from vissl.data.dataloader_sync_gpu_wrapper import DataloaderSyncGPUWrapper
from vissl.data.dataset_catalog import VisslDatasetCatalog, register_datasets
from vissl.data.disk_dataset import DiskImageDataset
from vissl.data.ssl_dataset import GenericSSLDataset
from vissl.data.synthetic_dataset import SyntheticImageDataset
from vissl.utils.misc import setup_multiprocessing_method


__all__ = [
    "GenericSSLDataset",
    "get_data_files",
    "register_datasets",
    "VisslDatasetCatalog",
]

DATASET_SOURCE_MAP = {
    "disk_filelist": DiskImageDataset,
    "disk_folder": DiskImageDataset,
    "synthetic": SyntheticImageDataset,
}


def build_dataset(cfg, split):
    """
    Given the user defined config and the dataset split (train/val), build
    the dataset.

    Args:
        cfg (AttrDict): user defined configuration file
        split (str): dataset split (from train or test)

    Returns:
        Instance of GenericSSLDataset
    """
    dataset = GenericSSLDataset(cfg, split, DATASET_SOURCE_MAP)
    return dataset


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


def get_sampler(dataset, dataset_config):
    """
    Given the dataset object and the dataset config, get the data sampler to use
    Supports 2 types of samplers:
        - Pytorch default torch.utils.data.distributed.DistributedSampler
        - VISSL sampler StatefulDistributedSampler that is written specifically for
          large scale dataset trainings
    """
    data_sampler = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if dataset_config["USE_STATEFUL_DISTRIBUTED_SAMPLER"]:
            data_sampler = StatefulDistributedSampler(
                dataset, batch_size=dataset_config["BATCHSIZE_PER_REPLICA"]
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


def get_loader(
    dataset: GenericSSLDataset,
    dataset_config: dict,
    num_dataloader_workers: int,
    pin_memory: bool,
    multi_processing_method: str,
    device: torch.device,
    get_sampler=get_sampler,
    worker_init_fn=None,
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
    data_sampler = get_sampler(dataset, dataset_config)
    collate_function = get_collator(
        dataset_config["COLLATE_FUNCTION"], dataset_config["COLLATE_FUNCTION_PARAMS"]
    )
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

    # If the targeted device is CUDA, set up async device copy:
    # - makes sure that samples are on device
    # - overlap the copy with the previous batch computation.
    if device.type == "cuda":
        if dataset.cfg["DATA"]["ENABLE_ASYNC_GPU_COPY"]:
            logging.info("Wrapping the dataloader to async device copies")  # NOQA
            dataloader = DataloaderAsyncGPUWrapper(dataloader)
        else:
            logging.info("Wrapping the dataloader to synchronous device copies")  # NOQA
            dataloader = DataloaderSyncGPUWrapper(dataloader)

    else:
        logging.warning("Selecting a CPU device")

    return dataloader
