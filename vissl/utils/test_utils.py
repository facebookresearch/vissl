# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from typing import List, Tuple

import torch
from vissl.config.attr_dict import AttrDict
from vissl.hooks import default_hook_generator
from vissl.utils.distributed_launcher import launch_distributed


@contextmanager
def in_temporary_directory(enabled: bool = True):
    """
    Context manager tocCreate a temporary direction and remove
    it at the end of the context
    """
    if enabled:
        temp_dir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir)
    else:
        yield os.getcwd()


def gpu_test(gpu_count: int = 1):
    """
    Annotation for GPU tests, skipping the test if the
    required amount of GPU is not available
    """
    import unittest

    message = f"Not enough GPUs to run the test: required {gpu_count}"
    return unittest.skipIf(torch.cuda.device_count() < gpu_count, message)


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

    def get_accuracies(self) -> List[str]:
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
