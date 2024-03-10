# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import torch
from classy_vision import tasks
from classy_vision.generic.distributed_util import is_primary
from classy_vision.hooks.classy_hook import ClassyHook

if is_primary():
    import wandb

BYTE_TO_MiB = 2 ** 20

class SSLWandbHook(ClassyHook):

    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_start = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop

    def __init__(
        self,
        log_params: bool = False,
        log_params_every_n_iterations: int = -1,
        log_params_gradients: bool = False,
    ) -> None:
        """The constructor method of SSLWandbHook.

        Args:
            log_params (bool): whether to log model params to wandb
            log_params_every_n_iterations (int): frequency at which parameters
                        should be logged to wandb
            log_params_gradients (bool): whether to log params gradients as well
                        to wandb.
        """
        super().__init__()
        # going to assume WandB install check is already performed (TODO: check this)

        logging.info("Setting up SSL Wandb Hook...")
        self.watched = False
        self.log_params = log_params
        self.log_params_every_n_iterations = log_params_every_n_iterations
        self.log_params_gradients = log_params_gradients
        logging.info(
            f"Wandb config: log_params: {self.log_params}, "
            f"log_params_freq: {self.log_params_every_n_iterations}, "
            f"log_params_gradients: {self.log_params_gradients}"
        )


    def on_forward(self, task: "tasks.ClassyTask") -> None:
        """
        Called after every forward if wandb hook is enabled.
        Logs the model parameters if the training iteration matches the
        logging frequency.
        """
        if not self.log_params:
            return

        if (
            self.log_params_every_n_iterations > 0
            and is_primary()
            and task.train
            and task.iteration % self.log_params_every_n_iterations == 0
        ):
            out_dict = {}
            for name, parameter in task.base_model.named_parameters():
                parameter = parameter.cpu().data.numpy()
                out_dict[f"Parameters/{name}"] = wandb.Histogram(parameter)

            wandb.log(out_dict, step=task.iteration)

    def on_phase_start(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the start of every epoch if the wandb hook is
        enabled.
        Logs the model parameters once at the beginning of training only.
        """
        if not self.log_params:
            return

        # log the parameters just once, before training starts
        if is_primary() and task.train and task.train_phase_idx == 0:
            out_dict = {}
            for name, parameter in task.base_model.named_parameters():
                parameter = parameter.cpu().data.numpy()
                out_dict[f"Parameters/{name}"] = wandb.Histogram(parameter)

            wandb.log(out_dict, step=task.iteration)

    def on_phase_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of every epoch if the wandb hook is
        enabled.
        Log model parameters and/or parameter gradients as set by user
        in the wandb configuration. Also resents the CUDA memory counter.
        """
        out_dict = {}

        # Log train/test accuracy
        if is_primary():
            phase_type = "Training" if task.train else "Testing"
            for meter in task.meters:
                if "accuracy" in meter.name:
                    for top_n, accuracies in meter.value.items():
                        for i, acc in accuracies.items():
                            tag_name = f"{phase_type}/Accuracy_" f" {top_n}_Output_{i}"
                            out_name[tag_name] = round(acc, 5)

        if not (self.log_params or self.log_params_gradients):
            if len(out_dict) > 0:
                wandb.log(out_dict, step=task.iteration)
            return

        if is_primary() and task.train:
            # Log the weights and bias at the end of the epoch
            if self.log_params:
                for name, parameter in task.base_model.named_parameters():
                    parameter = parameter.cpu().data.numpy()
                    out_dict[f"Parameters/{name}"] = wandb.Histogram(parameter)

            # Log the parameter gradients at the end of the epoch
            if self.log_params_gradients:
                for name, parameter in task.base_model.named_parameters():
                    if parameter.grad is not None:
                        try:
                            parameter = parameter.grad.cpu().data.numpy()
                            out_dict[f"Gradients/{name}"] = wandb.Histogram(parameter)
                        except ValueError:
                            logging.info(
                                f"Gradient histogram empty for {name}, "
                                f"iteration {task.iteration}. Unable to "
                                f"log gradient."
                            )

            # Reset the GPU Memory counter
            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()

            wandb.log(out_dict, step=task.iteration)


    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Called after every parameters update if wandb hook is enabled.
        Logs the parameter gradients if they are being set to log,
        log the scalars like training loss, learning rate, average training
        iteration time, batch size per gpu, img/sec/gpu, ETA, gpu memory used,
        peak gpu memory used.
        """

        if not is_primary():
            return

        out_dict = {}
        iteration = task.iteration

        if (
            self.log_params_every_n_iterations > 0
            and self.log_params_gradients
            and task.train
            and iteration % self.log_params_every_n_iterations == 0
        ):
            logging.info(f"Logging Parameter gradients. Iteration {iteration}")
            for name, parameter in task.base_model.named_parameters():
                if parameter.grad is not None:
                    try:
                        parameter = parameter.grad.cpu().data.numpy()
                        out_dict[f"Gradients/{name}"] = wandb.Histogram(parameter)
                    except ValueError:
                        logging.info(
                            f"Gradient histogram empty for {name}, "
                            f"iteration {task.iteration}. Unable to "
                            f"log gradient."
                        )

        if iteration % task.config["LOG_FREQUENCY"] == 0 or (
            iteration <= 100 and iteration % 5 == 0
        ):
            logging.info(f"Logging metrics. Iteration {iteration}")
            out_dict["Training/Loss"] = round(task.last_batch.loss.data.cpu().item(), 5)
            out_dict["Training/Learning_rate"] = round(task.optimizer.options_view.lr, 5)

            # Batch processing time
            if len(task.batch_time) > 0:
                batch_times = task.batch_time
            else:
                batch_times = [0]

            batch_time_avg_s = sum(batch_times) / max(len(batch_times), 1)
            out_dict["Speed/Batch_processing_time_ms"] = scalar_value=int(1000.0 * batch_time_avg_s)

            # Images per second per replica
            pic_per_batch_per_gpu = task.config["DATA"]["TRAIN"][
                "BATCHSIZE_PER_REPLICA"
            ]
            pic_per_batch_per_gpu_per_sec = (
                int(pic_per_batch_per_gpu / batch_time_avg_s)
                if batch_time_avg_s > 0
                else 0.0
            )
            out_dict["Speed/img_per_sec_per_gpu"] = pic_per_batch_per_gpu_per_sec

            # ETA
            avg_time = sum(batch_times) / len(batch_times)
            eta_secs = avg_time * (task.max_iteration - iteration)
            out_dict["Speed/ETA_hours"] = eta_secs / 3600.0

            # GPU Memory
            if torch.cuda.is_available():
                # Memory actually being used
                out_dict["Memory/Peak_GPU_Memory_allocated_MiB"] = \
                    torch.cuda.max_memory_allocated() / BYTE_TO_MiB

                # Memory reserved by PyTorch's memory allocator
                out_dict["Memory/Peak_GPU_Memory_reserved_MiB"] = \
                    torch.cuda.max_memory_reserved() / BYTE_TO_MiB  # byte to MiB

                out_dict["Memory/Current_GPU_Memory_reserved_MiB"] = \
                    torch.cuda.memory_reserved() / BYTE_TO_MiB  # byte to MiB

        if len(out_dict) > 0:
            wandb.log(out_dict, step=iteration)
