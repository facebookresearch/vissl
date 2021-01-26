# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import time

import torch
from classy_vision import tasks
from classy_vision.generic.profiler import (
    compute_activations,
    compute_flops,
    count_params,
)
from classy_vision.hooks.classy_hook import ClassyHook


class SSLModelComplexityHook(ClassyHook):
    """
    Logs the number of paramaters, forward pass FLOPs and activations of the model.
    Adapted from: https://github.com/facebookresearch/ClassyVision/blob/master/classy_vision/hooks/model_complexity_hook.py#L20    # NOQA
    """

    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def on_start(self, task) -> None:
        """
        Before the training starts, run one forward only pass of the model on the
        dummy input of shape specified by user in MODEL.MODEL_COMPLEXITY.INPUT_SHAPE
        We calculate the flops, activations and number of params in the model.
        """
        self.num_flops, self.num_activations, self.num_parameters = 0, 0, 0
        input_shape = task.config["MODEL"]["MODEL_COMPLEXITY"]["INPUT_SHAPE"]
        try:
            self.num_parameters = count_params(task.base_model)
            self.num_parameters = round(float(self.num_parameters) / 1000000, 4)
            try:
                self.num_flops = compute_flops(task.base_model, input_shape)
                if self.num_flops is None:
                    logging.info("FLOPs for forward pass: skipped.")
                self.num_flops = round(float(self.num_flops) / 1000000000, 4)
            except NotImplementedError:
                logging.warning(
                    "Unsupported modules found in model. FLOPs calculation skipped "
                )
                logging.debug("Exception: ", exc_info=True)
            try:
                self.num_activations = compute_activations(task.base_model, input_shape)
                self.num_activations = round(float(self.num_activations) / 1000000, 4)
            except NotImplementedError:
                logging.info("input_shape not found. Skipping activation calculation.")
            logging.info(
                f"#params (10^6): {self.num_parameters} "
                f"#FLOPs (10^9): {self.num_flops} "
                f"#activations (10^6): {self.num_activations}"
            )
        except Exception:
            logging.exception("Unexpected failure estimating model complexity.")


class SetDataSamplerEpochHook(ClassyHook):
    """
    We use DistributedDataSampler for sampling the data. At the beginnning of
    each training epoch/phase, we need to set the epoch for the sampler.
    """

    on_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop

    def on_phase_start(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the start of each epoch or phase to set the data
        sampler epoch. This is important to ensure the data
        is shuffled and the shuffling can be reproduced deterministically
        if the training is resumed from a checkpoint.
        """
        task.phase_start_time = time.time()
        task.batches = 0
        task.losses = []
        # (Re-)Shuffle data:
        phase_type = "train" if task.train else "test"
        # set epoch of distributed sampler
        if hasattr(task.dataloaders[phase_type], "sampler"):
            if hasattr(task.dataloaders[phase_type].sampler, "set_epoch"):
                # task.phase_idx is current running phase id
                task.dataloaders[phase_type].sampler.set_epoch(task.phase_idx)
        logging.info(f"Starting phase {task.phase_idx} [{phase_type}]")


class UpdateBatchesSeenHook(ClassyHook):
    """
    Book-keeping only hook. Tracks how many forward passes have been done.
    aka how many batches have been seen by the trainer irrespective of
    the train or test phase. updates task.batches
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop

    def on_forward(self, task: "tasks.ClassyTask") -> None:
        """
        Called each time forward pass is triggered. We update the number of batches
        we have seen. This is useful for debugging.
        """
        task.batches += 1


class UpdateTrainIterationNumHook(ClassyHook):
    """
    Book-keeping hook: updates the training iteration number (only updated
    if it's a training phase). task.iteration is updated.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop

    def on_forward(self, task: "tasks.ClassyTask") -> None:
        """
        Called each time forward pass is triggered. We update the number of batches
        we have seen. This is useful for debugging.
        """
        phase_type = "train" if task.train else "test"
        if phase_type == "train":
            task.iteration += 1


class UpdateTrainBatchTimeHook(ClassyHook):
    """
    After after parameters update step (training phase), we update the batch
    time aka the training time for the current iteration.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Called each time forward pass is triggered. We update the number of batches
        we have seen. This is useful for debugging.
        """
        if not task.train:
            return

        task.batch_time.append(time.time() - task.start_time)
        task.start_time = time.time()


class UpdateTestBatchTimeHook(ClassyHook):
    """
    Include the batch time for test phase as well and called every
    time loss has been computed. Only updates task.batch_time if
    it's a test phase and train phase is already updated by
    UpdateTrainBatchTimeHook hook.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop
    on_update = ClassyHook._noop

    def on_loss_and_meter(self, task: "tasks.ClassyTask") -> None:
        """
        Called each time a loss has been computed. Append the batch time for
        test phase.
        """
        # we call this here so that if we are running the test phase, we don't
        # have any update computed. But the loss only. So we call the update time
        # here to capture the loss for test as well.
        if not task.train:
            task.batch_time.append(time.time() - task.start_time)
            task.start_time = time.time()


class CheckNanLossHook(ClassyHook):
    """
    After every loss computation, verify the loss is not infinite.
    Called for both training/test phase.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop
    on_update = ClassyHook._noop

    def on_loss_and_meter(self, task: "tasks.ClassyTask") -> None:
        """
        Called each time a loss has been computed and checks that loss is
        not None.
        """
        # check the loss is not NaN. The loss is already all_reduced
        loss_val = task.last_batch.loss.data.cpu()
        if not torch.isfinite(loss_val).all():
            raise FloatingPointError(
                f"Infinite Loss or NaN at iteration={task.iteration}. Loss value: {loss_val}"
            )


class FreezeParametersHook(ClassyHook):
    """
    Hook that helps to freeze some specified model parameters for certain
    number of training iterations. The parameters
    are specified in a dictionary containing {param_name: frozen_iterations}.
    Used in SwAV training.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_update = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop

    def on_backward(self, task: "tasks.ClassyTask") -> None:
        """
        After every backward pass and before updating the parameters,
        check if there are parameters that should stay frozen.
        Set the grad to None for those params.
        """
        if len(task.config.MODEL.TEMP_FROZEN_PARAMS_ITER_MAP) == 0:
            return
        map_params_to_iters = {}
        for to_map in task.config.MODEL.TEMP_FROZEN_PARAMS_ITER_MAP:
            map_params_to_iters[to_map[0]] = to_map[1]
        for name, p in task.model.named_parameters():
            if (
                name in map_params_to_iters
                and task.iteration < map_params_to_iters[name]
            ):
                p.grad = None
