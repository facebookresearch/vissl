# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
All the hooks involved in human-readable logging
"""

import atexit
import datetime
import json
import logging
import time
from typing import List, Optional

import torch
from classy_vision import meters, tasks
from classy_vision.generic.distributed_util import get_rank, is_primary
from classy_vision.hooks.classy_hook import ClassyHook
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from iopath.common.file_io import g_pathmgr
from vissl.models.model_helpers import model_output_has_nan
from vissl.utils.checkpoint import CheckpointWriter, is_checkpoint_phase
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.io import save_file
from vissl.utils.logger import log_gpu_stats
from vissl.utils.perf_stats import PerfStats


class LogGpuMemoryHook(ClassyHook):
    """
    Hook executed at a specified iteration number and prints the
    memory summary for the primary device at several steps of training.
    """

    on_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(self, log_iteration_num: int = 1) -> None:
        super().__init__()
        self.log_iteration_num = log_iteration_num

    def on_phase_start(self, task: "tasks.ClassyTask") -> None:
        """
        Print the stats just before the training epoch starts
        """
        self._print_memory_summary(task, "on_phase_start")

    def on_forward(self, task: "tasks.ClassyTask") -> None:
        """
        Print the stats after the model forward pass is done
        """
        self._print_memory_summary(task, "on_forward")

    def on_backward(self, task: "tasks.ClassyTask") -> None:
        """
        Print the stats just after model.backward() is done
        """
        self._print_memory_summary(task, "on_backward")

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Print the stats just after model params are updated
        """
        self._print_memory_summary(task, "on_update")

    def _print_memory_summary(self, task: "tasks.ClassyTask", stage_name: str) -> None:
        if (
            is_primary()
            and (task.device.type == "cuda")
            and task.local_iteration_num == self.log_iteration_num
        ):
            logging.info(
                f"========= Memory Summary at {stage_name} ======="
                f"\n{torch.cuda.memory_summary()}\n"
            )


class DumpMemoryOnException(ClassyHook):
    """
    Hook that dumps the pytoch tensor in memory upon
    occurrence of an exception
    """

    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_update = ClassyHook._noop
    on_start = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop

    def __init__(self):
        super().__init__()
        self.dist_rank = get_machine_local_and_dist_rank()[1]

    def on_exception(self, task: "tasks.ClassyTask"):
        import gc

        iteration = task.local_iteration_num
        dump_name = f"memory_rank_{self.dist_rank}_dump_{iteration}.txt"
        with open(dump_name, "w") as f:
            for obj in gc.get_objects():
                try:
                    if self._is_pytorch(obj):
                        print(type(obj), obj.size(), file=f)
                except Exception:
                    pass
            print(torch.cuda.memory_summary(), file=f)

    @staticmethod
    def _is_pytorch(obj):
        return torch.is_tensor(obj) or (
            hasattr(obj, "data") and torch.is_tensor(obj.data)
        )


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

    def __init__(
        self, checkpoint_folder: str, btime_freq: Optional[int] = None
    ) -> None:
        """
        Args:
            checkpoint_folder: checkpoint directory where we will write the stdout.json
            btime_freq: if specified, logs average batch time of rolling_freq
                          batches also.
        """
        super().__init__()
        self.btime_freq: Optional[int] = btime_freq
        self.json_stdout_logger = None
        if is_primary():
            self.json_stdout_logger = g_pathmgr.open(
                f"{checkpoint_folder}/stdout.json",
                mode="a",
                buffering=1,  # line buffer - flush each line
            )
            atexit.register(self.json_stdout_logger.close)

    def on_start(self, task: "tasks.ClassyTask") -> None:
        """
        At the creation of the training loop, initialize a dictionary
        that can be used to set additional information to log at each
        epoch. This dictionary will be reset after each log.
        """
        task.additional_log_data = {}

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Executed after after parameter update. If the current phase is training,
        and it's a logging iteration, we compute and log several helpul training
        stats to keep track of ongoing training.

        For monitoring the batch size (average training iteration time), we allow
        monitoring the stats (optionally) for every N iterations to get better
        idea about the batch time and training eta.

        Set the btime_freq input using cfg.HOOKS.PERF_STATS.PERF_STAT_FREQUENCY=N
        ensuring that cfg.HOOKS.PERF_STATS.MONITOR_PERF_STATS = True.
        """
        if is_primary() and task.train:
            # Only log during training and on primary
            self._log_training_epoch(task)
        task.additional_log_data.clear()

    def _log_training_epoch(self, task):
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
            if isinstance(task.optimizer.options_view.lr, (set, list)):
                lr_val = list(task.optimizer.options_view.lr)
            else:
                lr_val = round(task.optimizer.options_view.lr, 5)
            if isinstance(task.optimizer.options_view.weight_decay, (set, list)):
                wd_val = list(task.optimizer.options_view.weight_decay)
            else:
                wd_val = round(task.optimizer.options_view.weight_decay, 5)
            batch_time = int(1000.0 * avg_time)
            rank = get_rank()

            log_data = {
                "Rank": rank,
                "ep": train_phase_idx,
                "iter": iteration,
                "lr": lr_val,
                "loss": loss_val,
                "btime(ms)": batch_time,
                "eta": eta_string,
                "peak_mem(M)": peak_mem_used,
                "weight_decay": wd_val,
            }

            # Add customized data registered by other hooks
            log_data.update(task.additional_log_data)

            if iteration == 1:
                # Set max iterations. Currently used in benchmark_suite_scheduler.py
                log_data["max_iterations"] = task.max_iteration

            if self.btime_freq and len(batch_times) >= self.btime_freq:
                rolling_avg_time = (
                    sum(batch_times[-self.btime_freq :]) / self.btime_freq
                )
                rolling_eta_secs = int(
                    rolling_avg_time * (task.max_iteration - iteration)
                )
                rolling_eta_str = str(datetime.timedelta(seconds=int(rolling_eta_secs)))
                rolling_btime = int(1000.0 * rolling_avg_time)
                log_data[f"btime({self.btime_freq}iters)(ms)"] = rolling_btime
                log_data["rolling_eta"] = rolling_eta_str

            # to maintain the backwards compatibility with the log.txt
            # logs, we convert the json to the previous format.
            # the stdout.json can be used to use the json format of logs.
            stdout_data = ""
            for key, value in log_data.items():
                stdout_data = (
                    f"{stdout_data}[{key}: {value}] "
                    if key == "ep"
                    else f"{stdout_data}{key}: {value}; "
                )
            logging.info(stdout_data.strip())
            self.json_stdout_logger.write(json.dumps(log_data) + "\n")


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

    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = world_size

    @classmethod
    def print_and_save_meters(
        cls,
        task: "tasks.ClassyTask",
        train_phase_idx: int,
        meters: List["meters.ClassyMeter"],
        metric_key_name_suffix: str = "",
    ):
        """
        Executed only on primary gpu at the end of each epoch. Computes the
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
        for meter in meters:
            if len(meters) > 0 and (
                (task.train and task.config["METERS"]["enable_training_meter"])
                or (not task.train)
            ):
                meter_value = meter.value
                metric_key = f"{phase_type}_{meter.name}"

                if metric_key_name_suffix:
                    metric_key = f"{metric_key}_{metric_key_name_suffix}"

                if metric_key not in task.metrics:
                    task.metrics[metric_key] = []
                task.metrics[metric_key].append(meter_value)
                save_metrics[metric_key] = meter_value
                logging.info(f"Rank: {rank}, name: {metric_key}, value: {meter_value}")
        meter_file = f"{checkpoint_folder}/metrics.json"
        save_file(save_metrics, meter_file, append_to_json=True)

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
        model_output = task.last_batch.model_output
        has_nan = model_output_has_nan(model_output)

        if has_nan:
            _, dist_rank = get_machine_local_and_dist_rank()
            logging.info(f"Infinite Model output or NaN at iteration={task.iteration}.")
            self.checkpoint_model(
                task,
                mode_frequency=1,
                mode_num=task.iteration,
                mode="iteration",
                world_size=self.world_size,
            )
            model_output_file = (
                f"{task.checkpoint_folder}/rank{dist_rank}_model_output.pth"
            )
            input_sample_file = (
                f"{task.checkpoint_folder}/rank{dist_rank}_input_sample.pth"
            )
            with g_pathmgr.open(model_output_file, "wb") as fwrite:
                torch.save(model_output, fwrite)
            with g_pathmgr.open(input_sample_file, "wb") as fwrite:
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
            self.checkpoint_model(
                task,
                mode_frequency=checkpoint_frequency,
                mode_num=task.iteration,
                mode="iteration",
                world_size=self.world_size,
            )

    def on_phase_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of each phase and forward. We log the metrics and also
        save the checkpoint. We pass the mode: phase or iteration
        """
        if is_primary():
            self.print_and_save_meters(task, task.train_phase_idx, task.meters)
        checkpoint_frequency = task.config["CHECKPOINT"]["CHECKPOINT_FREQUENCY"]
        self.checkpoint_model(
            task,
            world_size=self.world_size,
            mode_frequency=checkpoint_frequency,
            mode_num=task.train_phase_idx,
            mode="phase",
        )

    @staticmethod
    def checkpoint_model(
        task: "tasks.ClassyTask",
        world_size: int,
        mode_frequency: int,
        mode_num: int,
        mode: str = "phase",
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
        train_phase_idx = task.train_phase_idx
        # num_train_phases = num_epochs * num_phases_per_epoch
        # For OSS use, num_train_phases will be equal to num_epochs
        num_train_phases = task.num_train_phases

        # check if we need to checkpoint this phase
        is_checkpointing_phase = is_checkpoint_phase(
            mode_num, mode_frequency, train_phase_idx, num_train_phases, mode
        )
        is_final_train_phase = (
            (train_phase_idx == (num_train_phases - 1))
            and task.train
            and mode == "phase"
        )

        # handle checkpoint:
        if task.train and (is_final_train_phase or is_checkpointing_phase):
            #  - if sharded state consolidate the state
            # /!\ All the ranks have to participate
            if hasattr(task.optimizer, "consolidate_state_dict") and mode != "phase":
                logging.info(
                    f"[{mode}: {mode_num}] Consolidating sharded state on all replicas"
                )
                task.optimizer.consolidate_state_dict()

            # Depending on whether we are in FSDP mode or not
            # - save the checkpoint on the primary rank
            # - save the sharded checkpoint on all ranks
            if is_primary() or isinstance(task.base_model, FSDP):
                checkpoint_folder = task.checkpoint_folder
                logging.info(
                    f"[{mode}: {mode_num}] Saving checkpoint to {checkpoint_folder}"
                )
                model_state_dict = task.get_classy_state()

                # phase_idx is already incremented at the beginning of phase but if we
                # are checkpointing at an iteration in the middle of phase, we should not
                # save the incremented phase_idx as it will incorrectly assume that model
                # trained for that phase already.
                if mode == "iteration":
                    model_state_dict["phase_idx"] = model_state_dict["phase_idx"] - 1
                    if task.train:
                        train_phase_idx = train_phase_idx - 1
                        model_state_dict["train_phase_idx"] = train_phase_idx
                    restart_phase = phase_idx - 1
                    restart_iteration = task.iteration

                # When loading from a phase checkpoint:
                else:
                    restart_phase = phase_idx
                    restart_iteration = task.iteration

                if task.ema_model is not None:
                    model_state_dict["ema_model"] = (
                        task.ema_model.module.get_classy_state()
                    )

                    model_state_dict["ema_meters"] = [
                        meter.get_classy_state() for meter in task.ema_meters
                    ]

                # Content of the loss to be saved
                # - for DDP model, it contains the whole loss
                # - for FSDP model, loss can override state_dict() to
                #   save only the part related to the shard
                loss_state_dict = task.loss.state_dict()

                # Content of the checkpoint to be saved:
                # - for DDP model, it contains the whole model
                # - for FSDP model, it contains a shard of the model
                checkpoint_content = {
                    "phase_idx": restart_phase,
                    "iteration": restart_iteration,
                    "loss": loss_state_dict,
                    "iteration_num": task.local_iteration_num,
                    "train_phase_idx": train_phase_idx,
                    "classy_state_dict": model_state_dict,
                }

                # Saving the checkpoint:
                # - for DDP model, the primary save the whole model
                # - for FSDP model, each rank saves its own shard
                checkpoint_writer = CheckpointWriter(
                    checkpoint_folder=checkpoint_folder,
                    is_final_train_phase=is_final_train_phase,
                    mode=mode,
                    mode_num=mode_num,
                    backend=task.config["CHECKPOINT"]["BACKEND"],
                )
                if isinstance(task.base_model, FSDP):
                    _, rank = get_machine_local_and_dist_rank()
                    checkpoint_writer.save_sharded_checkpoint(
                        content=checkpoint_content,
                        shard_rank=rank,
                        world_size=world_size,
                    )
                else:
                    checkpoint_writer.save_consolidated_checkpoint(checkpoint_content)


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
