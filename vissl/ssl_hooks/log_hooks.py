# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
All the hooks involved in human-readable logging
"""

import datetime
import logging
import os
import time
from typing import Optional

import torch
from classy_vision import tasks
from classy_vision.generic.distributed_util import get_rank, is_master
from classy_vision.generic.util import save_checkpoint
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.utils.checkpoint import get_checkpoint_folder, is_checkpoint_phase
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.io import save_file
from vissl.utils.logger import log_gpu_stats
from vissl.utils.perf_stats import PerfStats


class LogGpuStatsHook(ClassyHook):
    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_update = ClassyHook._noop

    def on_start(self, task: "tasks.ClassyTask") -> None:
        if is_master() and task.use_gpu:
            # print the nvidia-smi stats
            log_gpu_stats()

    def on_step(self, task: "tasks.ClassyTask") -> None:
        # print the nvidia-smi stats again to get more accurate nvidia-smi
        # useful for monitoring memory usage.
        if is_master() and task.use_gpu and task.local_iteration_num == 50:
            log_gpu_stats()


class LogLossLrEtaHook(ClassyHook):
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop

    def __init__(self, btime_freq: Optional[int] = None) -> None:
        """
        Args:
            btime_freq: if specified, logs average batch time of rolling_freq
                          batches also.
        """
        super().__init__()
        self.btime_freq: Optional[int] = btime_freq

    def on_update(self, task: "tasks.ClassyTask") -> None:
        phase_type = "train" if task.train else "test"
        if is_master() and phase_type == "train":
            train_phase_idx = task.train_phase_idx
            log_freq = task.config["LOG_FREQUENCY"]
            iteration = task.iteration

            if torch.cuda.is_available():
                peak_mem_used = int(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
            else:
                peak_mem_used = -1

            if (
                (iteration == 1)
                or (iteration % log_freq == 0)
                or (iteration <= 100 and iteration % 5 == 0)
            ):
                loss_val = round(task.last_batch.loss.data.cpu().item(), 5)
                if len(task.batch_time) > 0:
                    batch_times = task.batch_time
                else:
                    batch_times = [0]
                avg_time = sum(batch_times) / len(batch_times)

                eta_secs = avg_time * (task.max_iteration - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_secs)))
                lr_val = round(task.optimizer.parameters.lr, 5)
                batch_time = int(1000.0 * avg_time)
                rank = get_rank()
                log_str = (
                    f"Rank: {rank}; "
                    f"[ep: {train_phase_idx}] "
                    f"iter: {iteration}; "
                    f"lr: {lr_val}; "
                    f"loss: {loss_val}; "
                    f"btime(ms): {batch_time}; "
                    f"eta: {eta_string}; "
                    f"peak_mem: {peak_mem_used}M"
                )
                if self.btime_freq and len(batch_times) >= self.btime_freq:
                    rolling_avg_time = (
                        sum(batch_times[-self.btime_freq :]) / self.btime_freq
                    )
                    rolling_eta_secs = int(
                        rolling_avg_time * (task.max_iteration - iteration)
                    )
                    rolling_eta_str = str(
                        datetime.timedelta(seconds=int(rolling_eta_secs))
                    )
                    rolling_btime = int(1000.0 * rolling_avg_time)
                    log_str = (
                        f"{log_str}; "
                        f"btime({self.btime_freq}iters): {rolling_btime} ms; "
                        f"rolling_eta: {rolling_eta_str}"
                    )
                logging.info(log_str)


class LogLossMetricsCheckpointHook(ClassyHook):
    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop

    def on_step(self, task: "tasks.ClassyTask") -> None:
        # in some cases, we might want to checkpoint after certain number of
        # iterations.
        checkpoint_frequency = task.config["CHECKPOINT"]["CHECKPOINT_ITER_FREQUENCY"]
        if checkpoint_frequency > 0:
            self._checkpoint_model(
                task,
                task.train_phase_idx,
                mode_frequency=checkpoint_frequency,
                mode_num=task.iteration,
                mode="iteration",
            )

    def on_phase_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of each phase and forward. We log the metrics and also
        save the checkpoint. We pass the mode: phase or iteration
        """
        if is_master():
            self._print_and_save_meters(task, task.train_phase_idx)
        checkpoint_frequency = task.config["CHECKPOINT"]["CHECKPOINT_FREQUENCY"]
        self._checkpoint_model(
            task,
            task.train_phase_idx,
            mode_frequency=checkpoint_frequency,
            mode_num=task.phase_idx,
            mode="phase",
        )

    def _checkpoint_model(
        self, task, train_phase_idx, mode_frequency, mode_num, mode="phase"
    ):
        phase_idx = task.phase_idx
        num_epochs = task.num_epochs
        # check if we need to checkpoint this phase
        is_checkpointing_phase = is_checkpoint_phase(
            mode_num, mode_frequency, train_phase_idx, num_epochs, mode
        )
        checkpoint_folder = get_checkpoint_folder(task.config)
        is_final_train_phase = (
            (train_phase_idx == (num_epochs - 1)) and task.train and mode == "phase"
        )
        # save checkpoint:
        if (
            is_master()
            and task.train
            and (checkpoint_folder is not None)
            and (is_final_train_phase or is_checkpointing_phase)
        ):
            logging.info(
                f"[{mode}: {mode_num}] Saving checkpoint to {checkpoint_folder}"
            )
            classy_state_dict = task.get_classy_state()
            # phase_idx is already incremented at the beginning of phase but if we
            # are checkpointing at an iteration in the middle of phase, we should not
            # save the incremented phase_idx as it will incorrectly assume that model
            # trained for that phase already.
            if mode == "iteration":
                phase_idx = phase_idx - 1
                classy_state_dict["phase_idx"] = classy_state_dict["phase_idx"] - 1
                if task.train:
                    train_phase_idx = train_phase_idx - 1
                    classy_state_dict["train_phase_idx"] = train_phase_idx
            checkpoint_task = {
                "phase_idx": phase_idx,
                "iteration": task.iteration,
                "criterion": task.loss.state_dict(),
                "iteration_num": task.local_iteration_num,
                "train_phase_idx": train_phase_idx,
                "classy_state_dict": classy_state_dict,
            }
            ckpt_name = f"model_{mode}{mode_num}.torch"
            if is_final_train_phase:
                ckpt_name = f"model_final_checkpoint_{mode}{mode_num}.torch"
            # TODO (prigoyal): add support for more backends: manifold etc.
            backend = task.config["CHECKPOINT"]["BACKEND"]
            assert backend == "disk", "Only disk BACKEND supported"
            save_checkpoint(
                checkpoint_folder, checkpoint_task, checkpoint_file=ckpt_name
            )
            logging.info(
                f"Saved checkpoint: {os.path.join(checkpoint_folder, ckpt_name)}"
            )

    def _print_and_save_meters(self, task, train_phase_idx):
        phase_type = "train" if task.train else "test"
        rank, _ = get_machine_local_and_dist_rank()
        checkpoint_folder = get_checkpoint_folder(task.config)
        save_metrics = {}
        save_metrics["iteration"] = task.iteration
        save_metrics["phase_idx"] = task.phase_idx
        save_metrics["train_phase_idx"] = train_phase_idx
        for meter in task.meters:
            metric_key = f"{phase_type}_{meter.name}"
            if metric_key not in task.metrics:
                task.metrics[metric_key] = []
            task.metrics[metric_key].append(meter.value)
            save_metrics[metric_key] = meter.value
            logging.info(f"Rank: {rank}, name: {metric_key}, value: {meter.value}")
        meter_file = os.path.join(checkpoint_folder, "metrics.json")
        save_file(save_metrics, meter_file)


class LogPerfTimeMetricsHook(ClassyHook):
    """
    Computes and prints performance metrics. Logs at the end of a phase
    or every log_freq if specified by user.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop

    def __init__(self, log_freq: Optional[int] = None) -> None:
        """
        Args:
            log_freq: if specified, logs every log_freq batches also.
        """
        super().__init__()
        self.log_freq: Optional[int] = log_freq
        self.start_time: Optional[float] = None

    def on_phase_start(self, task: "tasks.ClassyTask") -> None:
        """
        Initialize start time and reset perf stats
        """
        self.start_time = time.time()
        task.perf_stats = PerfStats()

    def on_loss_and_meter(self, task: "tasks.ClassyTask") -> None:
        """
        Log performance metrics every log_freq batches, if log_freq is not None.
        """
        if self.log_freq is None:
            return
        batches = len(task.losses)
        if batches and batches % self.log_freq == 0:
            self._log_performance_metrics(task)

    def on_phase_end(self, task: "tasks.ClassyTask") -> None:
        """
        Log performance metrics at the end of a phase if log_freq is None.
        """
        batches = len(task.losses)
        if batches:
            self._log_performance_metrics(task)

    def _log_performance_metrics(self, task: "tasks.ClassyTask") -> None:
        """
        Compute and log performance metrics.
        """
        phase_type = task.phase_type
        batches = len(task.losses)

        if self.start_time is None:
            logging.warning("start_time not initialized")
        else:
            # Average batch time calculation
            total_batch_time = time.time() - self.start_time
            average_batch_time = total_batch_time / batches
            logging.info(
                "Average %s batch time (ms) for %d batches: %d"
                % (phase_type, batches, 1000.0 * average_batch_time)
            )

        # Train step time breakdown
        if task.perf_stats is None:
            logging.warning('"perf_stats" not set in local_variables')
        elif task.train:
            logging.info(
                "Train step time breakdown (rank {}):\n{}".format(
                    get_rank(), task.perf_stats.report_str()
                )
            )
