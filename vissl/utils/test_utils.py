# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from vissl.config.attr_dict import AttrDict
from vissl.hooks import default_hook_generator
from vissl.utils.distributed_launcher import launch_distributed


@contextmanager
def in_temporary_directory(enabled: bool = True):
    """
    Context manager to create a temporary direction and remove
    it at the end of the context
    """
    if enabled:
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            yield temp_dir
            os.chdir(old_cwd)
    else:
        yield os.getcwd()


@contextmanager
def with_temp_files(count: int):
    """
    Context manager to create temporary files and remove them
    after at the end of the context
    """
    if count == 1:
        fd, file_name = tempfile.mkstemp()
        yield file_name
        os.close(fd)
    else:
        temp_files = [tempfile.mkstemp() for _ in range(count)]
        yield [t[1] for t in temp_files]
        for t in temp_files:
            os.close(t[0])


@dataclass
class TestTimer:
    elapsed_time_ms: int


@contextmanager
def with_timing(name: str):
    """
    Test utilities for basic performance tests
    """
    test_timer = TestTimer(elapsed_time_ms=0)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    yield test_timer
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    test_timer.elapsed_time_ms = start_event.elapsed_time(end_event)
    print(name, ":", test_timer.elapsed_time_ms, "ms")


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """
    import unittest

    message = f"Not enough GPUs to run the test: required {gpu_count}"
    return unittest.skipIf(torch.cuda.device_count() < gpu_count, message)


def init_distributed_on_file(world_size: int, gpu_id: int, sync_file: str):
    """
    Init the process group need to do distributed training, by syncing
    the different workers on a file.
    """
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(
        backend="nccl",
        init_method="file://" + sync_file,
        world_size=world_size,
        rank=gpu_id,
    )


def parse_losses_from_log_file(file_name: str):
    """
    Read a log file produced by VISSL and extract the losses produced
    at each iteration
    """
    iterations = []
    losses = []
    regex = re.compile(r"iter: (.*?); lr: (?:.*?); loss: (.*?);")
    with open(file_name, "r") as file:
        for line in file:
            if not line.startswith("INFO"):
                continue
            match = regex.search(line)
            if match is not None:
                iteration = int(match.group(1))
                loss = float(match.group(2))
                iterations.append(iteration)
                losses.append(loss)
    return iterations, losses


def parse_accuracies_from_log_file(file_name: str) -> List[str]:
    """
    Read a log file produced by VISSL and extract the list of accuracies
    as logged by VISSL (a string representation of a dictionary)
    """
    accuracies = []
    accuracy_tag = "accuracy_list_meter, value:"
    with open(file_name, "r") as file:
        for line in file:
            if accuracy_tag in line:
                i = line.index(accuracy_tag) + len(accuracy_tag)
                content = line[i:].strip()
                accuracies.append(content)
    return accuracies


class IntegrationTestLogs:
    """
    Helper function for integration tests, which provides common
    functions to read the output of the executed config
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir

    def clean_logs(self):
        log_path = os.path.join(self.run_dir, "log.txt")
        os.remove(log_path)

    def clean_final_checkpoint(self):
        for file_name in os.listdir(self.run_dir):
            if file_name.startswith("model_final_checkpoint_"):
                os.remove(file_name)

    def get_losses(self) -> List[float]:
        log_path = os.path.join(self.run_dir, "log.txt")
        return parse_losses_from_log_file(log_path)[1]

    def get_losses_with_iterations(self) -> Tuple[List[int], List[float]]:
        log_path = os.path.join(self.run_dir, "log.txt")
        return parse_losses_from_log_file(log_path)

    def get_accuracies(self, from_metrics_file: bool = False) -> List[str]:
        if from_metrics_file:
            metrics_path = os.path.join(self.run_dir, "metrics.json")
            with open(metrics_path, "r") as f:
                return [l.strip() for l in f]
        else:
            log_path = os.path.join(self.run_dir, "log.txt")
            return parse_accuracies_from_log_file(log_path)


def run_integration_test(
    config: AttrDict, engine_name: str = "train"
) -> IntegrationTestLogs:
    """
    Helper function to run an integration test on a given configuration
    """
    launch_distributed(
        cfg=config,
        node_id=0,
        engine_name=engine_name,
        hook_generator=default_hook_generator,
    )
    return IntegrationTestLogs(run_dir=os.getcwd())
