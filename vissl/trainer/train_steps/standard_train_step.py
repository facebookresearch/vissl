# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is the train step that"s most commonly used in most of the model trainings.
"""

import contextlib
from types import SimpleNamespace
from typing import Any, Dict

import torch
from classy_vision.generic.distributed_util import all_reduce_mean
from classy_vision.tasks import ClassyTask
from classy_vision.tasks.classification_task import AmpType
from vissl.hooks import SSLClassyHookFunctions
from vissl.trainer.train_steps import register_train_step
from vissl.utils.activation_checkpointing import (
    manual_gradient_all_reduce,
    manual_sync_params,
)
from vissl.utils.fsdp_utils import is_fsdp_model
from vissl.utils.misc import is_apex_available, torch_version
from vissl.utils.perf_stats import PerfTimer
from vissl.utils.profiler import record_function


if is_apex_available():
    import apex

# LastBatchInfo will typically hold
# the last samples, target, loss and output.
# More attributes can be added as needed in dependent codeblocks
LastBatchInfo = SimpleNamespace


@register_train_step("standard_train_step")
class StandardTrainStep:
    """
    Single training iteration loop of the model.

    Performs: data read, forward, loss computation, backward, optimizer step, parameter updates.

    Various intermediate steps are also performed:
    - logging the training loss, training eta, LR, etc to loggers
    - logging to tensorboard,
    - performing any self-supervised method specific operations (like in MoCo approach, the
    momentum encoder is updated), computing the scores in swav
    - checkpointing model if user wants to checkpoint in the middle
    of an epoch
    """

    def __call__(self, task: ClassyTask):
        assert isinstance(task, ClassyTask), "task is not instance of ClassyTask"

        # reset the last batch info at every step
        task.last_batch = LastBatchInfo()

        # We will time the training step and some of its sections, and accumulate values
        # into perf_stats if it were defined in local_variables:
        timer_train_step = PerfTimer("train_step_total", task.perf_stats)
        timer_train_step.start()

        # Create the next sample to process
        sample = self.construct_sample_for_model(task)

        # Only need gradients during training
        grad_context = torch.enable_grad() if task.train else torch.no_grad()
        ddp_context = self.create_ddp_context(task)
        torch_amp_context = self.create_amp_context(task)

        with grad_context, ddp_context, torch_amp_context:
            # Forward pass of the model
            with PerfTimer("forward", task.perf_stats), record_function("forward"):
                if task.enable_manual_gradient_reduction:
                    # Manually sync params and buffers for DDP.
                    manual_sync_params(task.model)
                model_output = task.model(sample["input"])

            # If the model outputs only one tensor, we take it out of the list.
            if len(model_output) == 1:
                model_output = model_output[0]

            task.last_batch.sample = sample
            task.last_batch.model_output = model_output
            target = sample["target"]

            # Run hooks on forward pass
            task.run_hooks(SSLClassyHookFunctions.on_forward.name)

            # Compute loss
            with PerfTimer("loss_compute", task.perf_stats), record_function(
                "loss_compute"
            ):
                local_loss = self.compute_loss(task, model_output, target)

            # Reduce the loss value across all nodes and gpus.
            with PerfTimer("loss_all_reduce", task.perf_stats):
                loss = local_loss.detach().clone()
                task.last_batch.loss = all_reduce_mean(loss)
            task.losses.append(task.last_batch.loss.data.cpu().item() * target.size(0))

            # Update the meters
            self.update_meters(task, model_output, target)

            task.last_batch.model_output = model_output
            task.last_batch.target = target

            # Update the iteration number, check loss is not NaN and measure batch time
            # now if it's a test phase since test phase doesn't have update step.
            task.run_hooks(SSLClassyHookFunctions.on_loss_and_meter.name)

        # Run backward now and update the optimizer
        if task.train:
            with PerfTimer("backward", task.perf_stats), record_function("backward"):
                self.run_backward(task, local_loss)
            task.run_hooks(SSLClassyHookFunctions.on_backward.name)
            with PerfTimer("optimizer_step", task.perf_stats), record_function(
                "optimizer_step"
            ):
                self.run_optimizer_step(task)

            task.run_hooks(SSLClassyHookFunctions.on_update.name)
            task.num_updates += task.get_global_batchsize()

        timer_train_step.stop()
        timer_train_step.record()
        return task

    @staticmethod
    def construct_sample_for_model(task: ClassyTask):
        """
        Given the input batch from the dataloader, verify the input is
        as expected: the input data and target data is present in the
        batch.
        In case of multi-input trainings like PIRL, make sure the data
        is in right format i.e. the multiple input should be nested
        under a common key "input".
        """
        # Process next sample
        with PerfTimer("read_sample", task.perf_stats):
            batch_data = next(task.data_iterator)

        sample_key_names = task.data_and_label_keys
        inp_key, target_key = sample_key_names["input"], sample_key_names["target"]
        all_keys = inp_key + target_key

        assert len(inp_key) + len(target_key) <= len(
            batch_data
        ), "Number of input and target keys in batch and train config don't match."

        # every input should be a list. The list corresponds to various data sources
        # and hence could be used to handle several data modalities.
        for key in all_keys:
            assert isinstance(batch_data[key], list), f"key: {key} input is not a list"
            assert (
                len(batch_data[key]) == 1
            ), f"Please modify your train step to handle multi-modal input: key {key}"

        # single input case
        if len(sample_key_names["input"]) == 1 and len(sample_key_names["target"]) == 1:
            sample = {
                "input": batch_data[inp_key[0]][0],
                "target": batch_data[target_key[0]][0],
                "data_valid": batch_data["data_valid"][0],
            }

        # multi-input case (example in PIRL, we pass image and patches both).
        # we nest all these under the sample["input"]
        elif len(sample_key_names["input"]) > 1:
            sample = {"input": {}, "target": {}, "data_valid": None}
            for key in inp_key:
                sample["input"][key] = batch_data[key][0]

            if len(target_key) > 1:
                for key in target_key:
                    sample["target"][key] = batch_data[key][0]
            else:
                sample["target"] = batch_data[target_key[0]][0]
            sample["data_valid"] = batch_data["data_valid"][0]

        # Copy the other keys as-is, method dependent
        # - But avoid to erase the standard keys
        for k in batch_data.keys():
            if k not in all_keys and k not in sample:
                sample[k] = batch_data[k]

        return sample

    @staticmethod
    def create_amp_context(task: ClassyTask):
        return (
            torch.cuda.amp.autocast()
            if task.amp_type == AmpType.PYTORCH
            else contextlib.suppress()
        )

    @staticmethod
    def create_ddp_context(task: ClassyTask):
        return (
            task.model.no_sync()
            if task.enable_manual_gradient_reduction
            else contextlib.suppress()
        )

    @classmethod
    def compute_loss(cls, task: ClassyTask, model_output, target):
        losses = task.loss(model_output, target)
        if isinstance(losses, dict):
            # For composite losses, log the sub-losses and extract the main one
            local_loss = losses.pop("loss")
            cls.log_sub_losses(task, losses)
        else:
            # Else assume that the loss is a single tensor
            assert isinstance(losses, torch.Tensor)
            local_loss = losses
        return local_loss

    @staticmethod
    def log_sub_losses(task: ClassyTask, sub_losses: Dict[str, Any]):
        for loss_key, loss_val in sub_losses.items():
            # If provided with a tensor, we assume that this is a local value
            # and perform the step to compute its mean across workers
            if isinstance(loss_val, torch.Tensor):
                loss_val = all_reduce_mean(loss_val.detach().clone()).item()
                task.additional_log_data[f"loss.{loss_key}"] = loss_val
            # Else, if provided with a scalar, we assume it has already
            # been computed appropriately
            elif isinstance(loss_val, (int, float)):
                task.additional_log_data[f"loss.{loss_key}"] = loss_val

    @staticmethod
    def update_meters(task: ClassyTask, model_output, target):
        if len(task.meters) == 0:
            return

        with_train_meters = task.config["METERS"]["enable_training_meter"]
        if (task.train and with_train_meters) or (not task.train):
            with PerfTimer("meters_update", task.perf_stats):
                if isinstance(model_output, list):
                    model_output_cpu = [x.cpu() for x in model_output]
                else:
                    model_output_cpu = model_output.cpu()
                for meter in task.meters:
                    meter.update(model_output_cpu, target.detach().cpu())

    @staticmethod
    def run_backward(task: ClassyTask, local_loss: torch.Tensor):
        task.optimizer.zero_grad()
        if task.amp_type == AmpType.APEX:
            with apex.amp.scale_loss(
                local_loss, task.optimizer.optimizer
            ) as scaled_loss:
                scaled_loss.backward()
                if task.enable_manual_gradient_reduction:
                    manual_gradient_all_reduce(task.model)

        elif task.amp_type == AmpType.PYTORCH:
            task.amp_grad_scaler.scale(local_loss).backward()
            if task.enable_manual_gradient_reduction:
                manual_gradient_all_reduce(task.model)
        else:
            local_loss.backward()
            if task.enable_manual_gradient_reduction:
                manual_gradient_all_reduce(task.model)

    @staticmethod
    def run_optimizer_step(task: ClassyTask):
        # Stepping the optimizer also updates learning rate, momentum etc
        # according to the schedulers (if any).
        assert task.where < 1.0, (
            "Optimizer being called with where=1.0. This should not happen "
            "as where=1.0 means training is already finished. Please debug your "
            "training setup. A common issue is the data sampler resuming "
            "where you are checkpointing model at every iterations but not using "
            "the stateful data sampler OR there's an issue in properly resuming the "
            "data sampler."
        )
        if task.amp_type == AmpType.PYTORCH:
            task.amp_grad_scaler.step(task.optimizer, where=task.where)
            task.amp_grad_scaler.update()
        else:
            task.optimizer.step(where=task.where)
        # set the model grads to None to save memory only in case of FSDP model
        if is_fsdp_model(task.model):
            zero_grad(task.model)


def zero_grad(model: torch.nn.Module) -> None:
    if torch_version() >= (1, 7, 0):
        model.zero_grad(set_to_none=True)
    else:
        for p in model.parameters():
            p.grad = None
