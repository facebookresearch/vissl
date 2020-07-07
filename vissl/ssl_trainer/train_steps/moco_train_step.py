# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This is the training process which was proposed in the "Momentum Contrast
for Unsupervised Visual Representation Learning" paper, from Kaiming He et al.
See http://arxiv.org/abs/1911.05722 for details
and https://github.com/facebookresearch/moco for a reference implementation

Although the structure from the original MoCo implementation is not completely maintained,
because of a different framework which encompasses more SSL training paradigms,
we try to follow it and use the same terminology wherever possible.
"""

import copy
from collections import namedtuple

import torch
from classy_vision.generic.distributed_util import all_reduce_mean
from classy_vision.tasks import ClassyTask
from torch import nn
from vissl.ssl_hooks import SSLClassyHookFunctions
from vissl.ssl_trainer.train_steps.standard_train_step import (
    LastBatchInfo,
    backward_and_optim_step,
    construct_sample_for_model,
)
from vissl.utils.perf_stats import PerfTimer


MocoSettings = namedtuple(
    "MocoSettings", "dim", "queue_size", "momentum", "temperature"
)
"""
MocoSettings

dim: feature dimension
queue_size: number of negative keys
momentum: moco momentum of updating key encoder
temperature: softmax temperature
"""

DEFAULT_MOCO_SETTINGS = MocoSettings(
    dim=128, queue_size=65536, momentum=0.999, temperature=0.07
)

# TODO: Handle SyncBatchNorm
# TODO: actually compute a loss


@torch.no_grad()
def _update_encoder(task: ClassyTask, momentum: float) -> None:
    """
    Momentum update of the key encoder:
    Each parameter becomes a weighted average of its old self and the
    newest encoder.
    """

    for param_q, param_k in zip(
        task.model.parameters(), task.moco_encoder.parameters()
    ):
        param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)


@torch.no_grad()
def _concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def _dequeue_and_enqueue(task: ClassyTask, keys: torch.Tensor, settings: MocoSettings):
    """Discard the oldest key from the MoCo queue, save the newest one,
    through a round-robin mechanism
    """
    # gather keys before updating queue
    keys = _concat_all_gather(keys)

    batch_size = keys.shape[0]

    ptr = int(task.moco_queue_ptr)
    assert settings.queue_size % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    task.moco_queue[:, ptr : ptr + batch_size] = keys.T
    ptr = (ptr + batch_size) % settings.queue_size  # move pointer

    task.moco_queue_ptr[0] = ptr


def _prepare_task(task: ClassyTask, settings: MocoSettings) -> None:
    # Create the encoder, which will slowly track the model
    task.moco_encoder = copy.deepcopy(task.model)

    # Create the queue
    task.moco_queue = torch.randn(settings.dim, settings.queue_size)
    task.moco_queue = nn.functional.normalize(task.moco_queue, dim=0)
    task.moco_queue_ptr = torch.zeros(1, dtype=torch.long)


def _compute_logits_labels(
    query: torch.Tensor, key: torch.Tensor, task: ClassyTask, settings: MocoSettings
):
    # TODO: @lefaudeux this is most probably wrong

    # Einstein sum is more intuitive
    # positive logits: Nx1
    l_pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)

    # negative logits: NxK
    l_neg = torch.einsum("nc,ck->nk", [query, task.moco_queue.clone().detach()])

    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= settings.temperature

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    return logits, labels


def _get_sample_partition(sample):
    # We need to split the positives in two partitions,
    # so that it goes to both the query and key codepaths

    # In MoCo's logic, there is no negative in a given batch
    # (as opposed to SimCLR for instance), the negatives come
    # from the previous keys

    # TODO:
    # - need to have only positives in the batch
    # - partition the positives so that it goes to the encoder and momentum_encoder
    # - return two batches which are properly aligned

    pass


def moco_train_step(
    task: ClassyTask, use_gpu: bool, settings: MocoSettings = DEFAULT_MOCO_SETTINGS
):
    assert isinstance(task, ClassyTask), "task is not instance of ClassyTask"

    # reset the last batch info at every step
    task.last_batch = {}

    if task.moco_encoder is None:
        _prepare_task(task, settings)

    # We'll time train_step and some of its sections, and accumulate values
    # into perf_stats if it were defined in local_variables:
    perf_stats = task.perf_stats
    timer_train_step = PerfTimer("train_step_total", perf_stats)
    timer_train_step.start()

    # -----------------------------------------------------------------
    # Process next sample
    with PerfTimer("read_sample", perf_stats):
        sample = next(task.data_iterator)
    sample = construct_sample_for_model(sample, task, use_gpu=use_gpu)
    target = sample["target"]  # TODO @lefaudeux this needs to be handled properly

    # Partition the inputs and target in between the query and key
    # (encoder vs. momentum encoder)

    # TODO @lefaudeux split the targets in between the query and key

    # Compute key features
    with torch.no_grad():
        # - update the encoder with momentum
        _update_encoder(task, settings.momentum)

        # - compute the new (normalized) key
        key = nn.functional.normalize(
            task.moco_encoder(sample["input_momentum_encoder"])
        )

        # - update the key queue
        _dequeue_and_enqueue(task, key, settings)

    # Compute query features.
    # gradient only needed during training
    context = torch.enable_grad() if task.train else torch.no_grad()

    with context:
        with PerfTimer("forward", perf_stats):
            query = task.model(sample["input"])

        # run hooks on forward pass
        task.run_hooks(SSLClassyHookFunctions.on_forward.name)

        # compute logits
        logits, labels = _compute_logits_labels(query, key, task, settings)

        # compute loss
        with PerfTimer("loss_compute", perf_stats):
            local_loss = task.loss(logits, labels)

        # Reduce the loss value across all nodes and gpus.
        with PerfTimer("loss_all_reduce", perf_stats):
            loss = local_loss.detach().clone()
            loss = all_reduce_mean(loss)

        task.losses.append(loss.data.cpu().item() * labels.size(0))

        # update meters
        if len(task.meters) > 0:
            with PerfTimer("meters_update", perf_stats):
                for meter in task.meters:
                    meter.update(logits.cpu(), labels.detach().cpu())

        # create the LastBatchInfo object with the current batch info
        task.last_batch = LastBatchInfo(
            loss=loss, output=logits, target=labels, sample=sample
        )

        # update the iteration number, check loss is not NaN and measure batch time
        # now if it's a test phase since test phase doesn't have update step.
        task.run_hooks(SSLClassyHookFunctions.on_loss_and_meter.name)

    # run backward now and update the optimizer
    if task.train:
        backward_and_optim_step(task, local_loss)

    timer_train_step.stop()
    timer_train_step.record()

    return task
