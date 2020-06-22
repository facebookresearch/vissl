#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This is the train step that"s most commonly used in most of the model trainings.
"""

from typing import Any, Dict, NamedTuple

import torch
from classy_vision.generic.distributed_util import all_reduce_mean
from classy_vision.generic.util import recursive_copy_to_gpu
from classy_vision.tasks import ClassyTask
from vissl.ssl_hooks import SSLClassyHookFunctions
from vissl.utils.misc import is_apex_available
from vissl.utils.perf_stats import PerfTimer


if is_apex_available():
    import apex


class LastBatchInfo(NamedTuple):
    loss: torch.Tensor
    output: torch.Tensor
    target: torch.Tensor
    sample: Dict[str, Any]


def construct_sample_for_model(batch_data, task):
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
        ), "Please modify your train step to handle multi-modal input"

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
    return sample


def standard_train_step(task, use_gpu):  # NOQA
    assert isinstance(task, ClassyTask), "task is not instance of ClassyTask"

    # reset the last batch info at every step
    task.last_batch = {}

    # We'll time train_step and some of its sections, and accumulate values
    # into perf_stats if it were defined in local_variables:
    perf_stats = task.perf_stats
    timer_train_step = PerfTimer("train_step_total", perf_stats)
    timer_train_step.start()

    # Process next sample
    with PerfTimer("read_sample", perf_stats):
        sample = next(task.data_iterator)
    sample = construct_sample_for_model(sample, task)

    # copy sample to GPU recursively
    if use_gpu:
        for key, value in sample.items():
            sample[key] = recursive_copy_to_gpu(value, non_blocking=True)

    # Only need gradients during training
    context = torch.enable_grad() if task.train else torch.no_grad()
    with context:
        # Forward pass of the model
        with PerfTimer("forward", perf_stats):
            model_output = task.model(sample["input"])
        target = sample["target"]

        # run hooks on forward pass
        task.run_hooks(SSLClassyHookFunctions.on_forward.name)

        # compute loss
        with PerfTimer("loss_compute", perf_stats):
            local_loss = task.loss(model_output, target)

        # Reduce the loss value across all nodes and gpus.
        with PerfTimer("loss_all_reduce", perf_stats):
            loss = local_loss.detach().clone()
            loss = all_reduce_mean(loss)

        task.losses.append(loss.data.cpu().item() * target.size(0))

        # update meters
        if len(task.meters) > 0:
            with PerfTimer("meters_update", perf_stats):
                if isinstance(model_output, list):
                    if use_gpu:
                        model_output_cpu = [x.cpu() for x in model_output]
                    else:
                        model_output_cpu = model_output
                else:
                    model_output_cpu = model_output.cpu() if use_gpu else model_output
                for meter in task.meters:
                    meter.update(model_output_cpu, target.detach().cpu())

        # create the LastBatchInfo object with the current batch info
        task.last_batch = LastBatchInfo(
            loss=loss, output=model_output, target=target, sample=sample
        )
        # update the iteration number, check loss is not NaN and measure batch time
        # now if it's a test phase since test phase doesn't have update step.
        task.run_hooks(SSLClassyHookFunctions.on_loss_and_meter.name)

    # run backward now and update the optimizer
    if task.train:
        with PerfTimer("backward", perf_stats):
            task.optimizer.zero_grad()
            if task.amp_args is not None and is_apex_available():
                with apex.amp.scale_loss(
                    local_loss, task.optimizer.optimizer
                ) as scaled_loss:
                    scaled_loss.backward()
            else:
                local_loss.backward()

        task.optimizer.update_schedule_on_step(task.where)
        task.run_hooks(SSLClassyHookFunctions.on_backward.name)
        with PerfTimer("optimizer_step", perf_stats):
            task.optimizer.step()
        task.run_hooks(SSLClassyHookFunctions.on_update.name)
        task.num_updates += task.get_global_batchsize()

    timer_train_step.stop()
    timer_train_step.record()

    return task
