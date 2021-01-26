# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
All the hooks involved in human-readable logging
"""

import datetime
import logging
import time
from typing import Optional

import torch
from classy_vision import tasks
from classy_vision.generic.distributed_util import get_rank, is_primary
from classy_vision.generic.util import save_checkpoint
from classy_vision.hooks.classy_hook import ClassyHook
from fvcore.common.file_io import PathManager
from vissl.utils.checkpoint import is_checkpoint_phase
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.io import create_file_symlink, save_file
from vissl.utils.logger import log_gpu_stats
from vissl.utils.perf_stats import PerfStats


class LogGpuStatsHook(ClassyHook):
    """
    Hook executed at the start of training and after every training iteration is done.
    Logs Gpu nvidia-smi stats to logger streams: at the start of training and
    after 50 training iterations.
    """

    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_update = ClassyHook._noop

    def on_start(self, task: "tasks.ClassyTask") -> None:
        """
        Logs Gpu nvidia-smi stats to logger streams.
        """
        if is_primary() and (task.device.type == "cuda"):
            # print the nvidia-smi stats
            log_gpu_stats()

    def on_step(self, task: "tasks.ClassyTask") -> None:
        """
        Print the nvidia-smi stats again to get more accurate nvidia-smi
        useful for monitoring memory usage.
        """
        if (
            is_primary()
            and (task.device.type == "cuda")
            and task.local_iteration_num == 50
        ):
            log_gpu_stats()


class LogLossLrEtaHook(ClassyHook):
    """
    Hook executed after every parameters update step. Logs training
    stats like: LR, iteration, ETA, batch time etc to logger streams.
    """

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
        """
        Executed after after parameter update. If the current phase is training,
        and it's a logging iteration, we compute and log several helpul training
        stats to keep track of ongoing training.

        For monitoring the batch size (average training iteration time), we allow
        monitoring the stats (optionally) for every N iterations to get better
        idea about the batch time and training eta.

        Set the btime_freq input using cfg.PERF_STAT_FREQUENCY=N ensuring that
        cfg.MONITOR_PERF_STATS = True.
        """
        phase_type = "train" if task.train else "test"
        if is_primary() and phase_type == "train":
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
                if isinstance(task.optimizer.options_view.lr, set):
                    lr_val = list(task.optimizer.options_view.lr)
                else:
                    lr_val = round(task.optimizer.options_view.lr, 5)
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
    """
    Hook called after every forward pass (to check training doesn't give NaN),
    after every step and at the end of epoch (to check if the model should be checkpointed)
    and print the meters values at the end of every phase.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop

    # TODO: make this a standalone hook and make it optional to save runtime
    # although the overhead is minimal when the model is training fine (no nans)
    def on_forward(self, task: "tasks.ClassyTask") -> None:
        """
        Called each time a model forward is done and make sure that
        the model forward output is not NaN. If we encounter NaN as the model
        output, we checkpoint the model to enable debugging and also checkpoint
        the model input sample, model output.
        """
        # check the model output is not NaN.
        has_nan = False
        model_output = task.last_batch.model_output
        if isinstance(model_output, list):
            has_nan = not torch.tensor(
                [torch.isfinite(x).all() for x in model_output]
            ).all()
        else:
            has_nan = not torch.isfinite(model_output).all()

        if has_nan:
            _, dist_rank = get_machine_local_and_dist_rank()
            logging.info(f"Infinite Model output or NaN at iteration={task.iteration}.")
            self._checkpoint_model(
                task,
                task.train_phase_idx,
                mode_frequency=1,
                mode_num=task.iteration,
                mode="iteration",
            )
            model_output_file = (
                f"{task.checkpoint_folder}/rank{dist_rank}_model_output.pth"
            )
            input_sample_file = (
                f"{task.checkpoint_folder}/rank{dist_rank}_input_sample.pth"
            )
            with PathManager.open(model_output_file, "wb") as fwrite:
                torch.save(model_output, fwrite)
            with PathManager.open(input_sample_file, "wb") as fwrite:
                torch.save(task.last_batch.sample, fwrite)
            logging.info(f"Saved model output: {model_output_file}")
            logging.info(f"Saved model input: {input_sample_file}")

    def on_step(self, task: "tasks.ClassyTask") -> None:
        """
        In some cases, we might want to checkpoint after certain number of iterations.
        If we want to checkpoint after every N iterations, check the checkpoint
        frequency matches and checkpoint if it does.
        """
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
        if is_primary():
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
        """
        Checkpoint model. Can be called in 3 possible scenarios:
        1. If training becomes NaN, then we checkpoint the model to facilitate debugging
        2. After every N epochs (CHECKPOINT_FREQ), model state is checkpointed.
        3. If user wants to checkpoint during the epoch (ie. after every few training
           iterations, the model state is checkpointed.)

        Args:
            task: Self-supervision task that hold information about training iteration,
                  epoch number etc.
            train_phase_idx (int): current training phase number. Starts from 0
            mode_frequency (int): mode can be "phase" or "iteration". Frequency
                                  of checkpointing for the given mode
            mode_num (int): for the checkpointing mode (phase or iteration), the number
                            of phase or iteration at which checkpointing is being done
        """
        phase_idx = task.phase_idx
        num_epochs = task.num_epochs
        # check if we need to checkpoint this phase
        is_checkpointing_phase = is_checkpoint_phase(
            mode_num, mode_frequency, train_phase_idx, num_epochs, mode
        )
        is_final_train_phase = (
            (train_phase_idx == (num_epochs - 1)) and task.train and mode == "phase"
        )
        # save checkpoint:
        if (
            is_primary()
            and task.train
            and (is_final_train_phase or is_checkpointing_phase)
        ):
            checkpoint_folder = task.checkpoint_folder
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
                "loss": task.loss.state_dict(),
                "iteration_num": task.local_iteration_num,
                "train_phase_idx": train_phase_idx,
                "classy_state_dict": classy_state_dict,
            }
            ckpt_name = f"model_{mode}{mode_num}.torch"
            if is_final_train_phase:
                ckpt_name = f"model_final_checkpoint_{mode}{mode_num}.torch"
            backend = task.config["CHECKPOINT"]["BACKEND"]
            assert backend == "disk", "Only disk BACKEND supported"
            save_checkpoint(
                checkpoint_folder, checkpoint_task, checkpoint_file=ckpt_name
            )
            logging.info(f"Saved checkpoint: {checkpoint_folder}/{ckpt_name}")
            # we create the checkpoint symlink and use this symlink to load
            # checkpoints. This helps ensure that the checkpoint we load from
            # are valid. It's a particularly useful feature for resuming trainings.
            logging.info("Creating symlink...")
            symlink_dest_file = f"{checkpoint_folder}/checkpoint.torch"
            source_file = f"{checkpoint_folder}/{ckpt_name}"
            create_file_symlink(source_file, symlink_dest_file)
            logging.info(f"Created symlink: {symlink_dest_file}")

    def _print_and_save_meters(self, task, train_phase_idx):
        """
        Executed only on master gpu at the end of each epoch. Computes the
        meters and logs the metrics to the json file and to logger streams
        (stdout, file).
        """
        phase_type = "train" if task.train else "test"
        rank, _ = get_machine_local_and_dist_rank()
        checkpoint_folder = task.checkpoint_folder
        save_metrics = {}
        save_metrics["iteration"] = task.iteration
        save_metrics["phase_idx"] = task.phase_idx
        save_metrics["train_phase_idx"] = train_phase_idx
        for meter in task.meters:
            if len(task.meters) > 0 and (
                (task.train and task.config["METERS"]["enable_training_meter"])
                or (not task.train)
            ):
                meter_value = meter.value
                metric_key = f"{phase_type}_{meter.name}"
                if metric_key not in task.metrics:
                    task.metrics[metric_key] = []
                task.metrics[metric_key].append(meter_value)
                save_metrics[metric_key] = meter_value
                logging.info(f"Rank: {rank}, name: {metric_key}, value: {meter_value}")
        meter_file = f"{checkpoint_folder}/metrics.json"
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
