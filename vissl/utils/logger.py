# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import functools
import logging
import os
import subprocess
import sys

from fvcore.common.file_io import PathManager
from vissl.utils.io import makedir


def setup_logging(name, output_dir=None, rank=0):
    # get the filename if we want to log to the files as well
    log_filename = None
    if output_dir:
        makedir(output_dir)
        log_filename = f"{output_dir}/log.txt"
        if rank > 0:
            log_filename = f"{log_filename}.rank{rank}"
        PathManager.mkdirs(os.path.dirname(log_filename))

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # clean up any pre-existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # we log to file as well if user wants
    if log_filename:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


def shutdown_logging():
    logging.info("Shutting down loggers...")
    handlers = logging.root.handlers
    for handler in handlers:
        handler.close()


def log_gpu_stats():
    logging.info(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))


def print_gpu_memory_usage():
    sp = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split("\n")
    all_values, count, out_dict = [], 0, {}
    for item in out_list:
        if " MiB" in item:
            out_dict[f"GPU {count}"] = item.strip()
            all_values.append(int(item.split(" ")[0]))
            count += 1
    logging.info(
        f"Memory usage stats:\n"
        f"Per GPU mem used: {out_dict}\n"
        f"nMax memory used: {max(all_values)}"
    )
