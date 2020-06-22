#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import torch
from classy_vision import tasks
from classy_vision.generic.distributed_util import is_master
from classy_vision.hooks.classy_hook import ClassyHook


try:
    from torch.utils.tensorboard import SummaryWriter  # noqa F401

    tb_available = True
except ImportError:
    # Make sure that the type hint is not blocking
    # on a non-TensorBoard aware platform
    from typing import TypeVar

    SummaryWriter = TypeVar("SummaryWriter")
    tb_available = False

BYTE_TO_MiB = 2 ** 20


class SSLTensorboardHook(ClassyHook):
    """
    SSL Specific variant of the Classy Vision tensorboard hook
    """

    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_start = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop

    def __init__(self, tb_writer: SummaryWriter, log_activations: bool = False) -> None:
        """The constructor method of SSLTensorboardHook.

        Args:
            tb_writer: `Tensorboard SummaryWriter <https://tensorboardx.
            readthedocs.io/en/latest/tensorboard.html#tensorboardX.
            SummaryWriter>`_ instance

        """
        super().__init__()
        if not tb_available:
            raise RuntimeError(
                "tensorboard not installed, cannot use SSLTensorboardHook"
            )
        logging.info("Setting up SSL Tensorboard Hook...")
        self.tb_writer = tb_writer
        self.log_activations = log_activations

    def on_phase_start(self, task: "tasks.ClassyTask") -> None:
        if not self.log_activations:
            return

        # log the parameters just once, before training starts
        if is_master() and task.train and task.train_phase_idx == 0:
            for name, parameter in task.base_model.named_parameters():
                self.tb_writer.add_histogram(
                    f"Parameters/{name}", parameter, global_step=-1
                )

    def on_phase_end(self, task: "tasks.ClassyTask") -> None:
        if not self.log_activations:
            return

        # Log the weights and bias at the end of the epoch
        if is_master() and task.train:
            for name, parameter in task.base_model.named_parameters():
                self.tb_writer.add_histogram(
                    f"Parameters/{name}", parameter, global_step=task.train_phase_idx
                )

            # Reset the GPU Memory counter
            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()

    def on_update(self, task: "tasks.ClassyTask") -> None:
        if not is_master():
            return

        iteration = task.iteration

        if iteration % task.config["LOG_FREQUENCY"] == 0 or (
            iteration <= 100 and iteration % 5 == 0
        ):
            logging.info(f"Logging metrics. Iteration {iteration}")
            self.tb_writer.add_scalar(
                tag="Loss",
                scalar_value=round(task.last_batch.loss.data.cpu().item(), 5),
                global_step=iteration,
            )

            self.tb_writer.add_scalar(
                tag="Learning_rate",
                scalar_value=round(task.optimizer.parameters.lr, 5),
                global_step=iteration,
            )

            # Batch processing time
            if len(task.batch_time) > 0:
                batch_times = task.batch_time
            else:
                batch_times = [0]

            self.tb_writer.add_scalar(
                tag="Batch_processing_time_ms",
                scalar_value=int(1000.0 * sum(batch_times) / len(batch_times)),
                global_step=iteration,
            )

            # ETA
            avg_time = sum(batch_times) / len(batch_times)
            eta_secs = avg_time * (task.max_iteration - iteration)
            self.tb_writer.add_scalar(
                tag="ETA_hours", scalar_value=eta_secs / 3600.0, global_step=iteration
            )

            # GPU Memory
            if torch.cuda.is_available():
                # Memory actually being used
                self.tb_writer.add_scalar(
                    tag="Peak_GPU_Memory_allocated_MiB",
                    scalar_value=torch.cuda.max_memory_allocated() / BYTE_TO_MiB,
                    global_step=iteration,
                )

                # Memory reserved by PyTorch's memory allocator
                self.tb_writer.add_scalar(
                    tag="Peak_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.max_memory_reserved()
                    / BYTE_TO_MiB,  # byte to MiB
                    global_step=iteration,
                )

                self.tb_writer.add_scalar(
                    tag="Current_GPU_Memory_reserved_MiB",
                    scalar_value=torch.cuda.memory_reserved()
                    / BYTE_TO_MiB,  # byte to MiB
                    global_step=iteration,
                )
