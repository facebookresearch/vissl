# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import sys
from typing import List

import torch
import tqdm
from fvcore.common.timer import Timer
from hydra.experimental import compose, initialize_config_module
from vissl.data import build_dataset, get_loader
from vissl.utils.hydra_config import AttrDict, convert_to_attrdict, is_hydra_available
from vissl.utils.logger import setup_logging


WARMUP_ITERS = 10
MAX_ITERS = 500
BENCHMARK_ROUNDS = 2


def benchmark_data(cfg: AttrDict, split: str = "train"):
    split = split.upper()
    total_images = MAX_ITERS * cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
    timer = Timer()
    dataset = build_dataset(cfg, split)

    try:
        device = torch.device("cuda" if cfg.MACHINE.DEVICE == "gpu" else "cpu")
    except AttributeError:
        device = torch.device("cuda")

    dataloader = get_loader(
        dataset=dataset,
        dataset_config=cfg["DATA"][split],
        num_dataloader_workers=cfg.DATA.NUM_DATALOADER_WORKERS,
        pin_memory=False,
        multi_processing_method=cfg.MULTI_PROCESSING_METHOD,
        device=device,
    )

    # Fairstore data sampler would require setting the start iter before it can start.
    if hasattr(dataloader.sampler, "set_start_iter"):
        dataloader.sampler.set_start_iter(0)

    # initial warmup measured as warmup time
    timer.reset()
    data_iterator = iter(dataloader)
    for i in range(10):  # warmup
        next(data_iterator)
        if i == 0:
            # the total number of seconds since the start/reset of the timer
            warmup_time = timer.seconds()
    logging.info(f"Warmup time {WARMUP_ITERS} batches: {warmup_time} seconds")

    # measure the number of images per sec in 1000 iterations.
    timer = Timer()
    for _ in tqdm.trange(MAX_ITERS):
        next(data_iterator)
    time_elapsed = timer.seconds()
    logging.info(
        f"iters: {MAX_ITERS}; images: {total_images}; time: {time_elapsed} seconds; "
        f"images/sec: {round(float(total_images / time_elapsed), 4)}; "
        f"ms/img: {round(float(1000 * time_elapsed / total_images), 4)} "
    )

    # run benchmark for a few more rounds to catch fluctuations
    for round_idx in range(BENCHMARK_ROUNDS):
        timer = Timer()
        for _ in tqdm.trange(MAX_ITERS):
            next(data_iterator)
        time_elapsed = timer.seconds()
        logging.info(
            f"round: {round_idx}: iters: {MAX_ITERS}; images: {total_images}; "
            f"time: {time_elapsed} seconds; "
            f"images/sec: {round(float(total_images / time_elapsed), 4)}; "
            f"ms/img: {round(float(1000 * time_elapsed / total_images), 4)} "
        )
    del data_iterator
    del dataloader


def hydra_main(overrides: List[str]):
    print(f"####### overrides: {overrides}")
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    setup_logging(__name__)
    args, config = convert_to_attrdict(cfg)
    benchmark_data(config)


if __name__ == "__main__":
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    overrides.append("hydra.verbose=true")
    hydra_main(overrides=overrides)
