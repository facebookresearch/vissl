# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import torch
from torch.utils.data import DataLoader
from vissl.data.collators import get_collator
from vissl.data.data_helper import StatefulDistributedSampler
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
    dataset = GenericSSLDataset(cfg, split, DATASET_SOURCE_MAP)
    return dataset


def print_sampler_config(data_sampler):
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
    logging.info("Distributed Sampler config:\n{}".format(sampler_cfg))


def get_sampler_fn(dataset, dataset_config):
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
    dataset,
    dataset_config,
    num_dataloader_workers,
    pin_memory,
    multi_processing_method,
    get_sampler_fn=get_sampler_fn,
):
    # pytorch dataloader requires setting the multiprocessing type.
    setup_multiprocessing_method(multi_processing_method)
    # we don't need to set the rank, replicas as the Sampler already does so in
    # it's init function
    data_sampler = get_sampler_fn(dataset, dataset_config)
    collate_function = get_collator(dataset_config["COLLATE_FUNCTION"])
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=num_dataloader_workers,
        pin_memory=pin_memory,
        shuffle=False,
        batch_size=dataset_config["BATCHSIZE_PER_REPLICA"],
        collate_fn=collate_function,
        sampler=data_sampler,
        drop_last=dataset_config["DROP_LAST"],
    )
    return dataloader
