# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from fvcore.common.file_io import PathManager
from vissl.data import AirstoreDataset, GenericSSLDataset
from vissl.models.model_helpers import model_output_has_nan
from vissl.utils.env import get_machine_local_and_dist_rank


class SSLModelComplexityHook(ClassyHook):
    """
    Logs the number of paramaters, forward pass FLOPs and activations of the model.
    Adapted from: https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/hooks/model_complexity_hook.py#L20    # NOQA
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
        dummy input of shape specified by user in HOOKS.MODEL_COMPLEXITY.INPUT_SHAPE
        We calculate the flops, activations and number of params in the model.
        """
        self.num_flops, self.num_activations, self.num_parameters = 0, 0, 0
        input_shape = task.config["HOOKS"]["MODEL_COMPLEXITY"]["INPUT_SHAPE"]
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

        # call set_epoch and for AirstoreDataset since it handles shuffle
        # behavior internally
        if hasattr(task.dataloaders[phase_type], "dataset"):
            dataset = task.dataloaders[phase_type].dataset
            if isinstance(dataset, GenericSSLDataset):
                for data_obj in dataset.data_objs:
                    if isinstance(data_obj, AirstoreDataset):
                        # task.phase_idx is current running phase id
                        data_obj.set_epoch(task.phase_idx)

        logging.info(f"Starting phase {task.phase_idx} [{phase_type}]")


class CheckNanModelOutputHook(ClassyHook):
    """
    After every model forward, verify the loss is not infinite.
    Called for both training/test phase.
    """

    on_start = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_step = ClassyHook._noop
    on_update = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop

    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = world_size

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

            # TODO - this code was broken during a refactoring: improve it
            from vissl.hooks.log_hooks import LogLossMetricsCheckpointHook

            LogLossMetricsCheckpointHook.checkpoint_model(
                task,
                world_size=self.world_size,
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

        # get the maximum iterations until which the params are frozen.
        # if the iterations are past the maximum iterations freezing any
        # param, we simply return.
        max_iterations = max(list(map_params_to_iters.values()))
        if task.iteration >= max_iterations:
            if task.iteration == max_iterations:
                logging.info(
                    f"No parameters grad removed from now on: {task.iteration}"
                )
            return

        num_matched_named_params = 0
        for name, p in task.model.named_parameters():
            match_param_name = self._clean_param_path(name)
            if (
                match_param_name in map_params_to_iters
            ) and task.iteration < map_params_to_iters[match_param_name]:
                num_matched_named_params += 1
                p.grad = None

        # TODO (Min): we need to check the exact target number.
        assert num_matched_named_params > 0, (
            f"Didn't find expected number of layers: "
            f"{num_matched_named_params} vs.  {len(map_params_to_iters)}"
        )

    @staticmethod
    def _clean_param_path(param_name: str) -> str:
        # Remove FSDP path artifacts
        paths_to_remove = ["_fsdp_wrapped_module.", "_fpw_module."]
        for path_to_remove in paths_to_remove:
            param_name = param_name.replace(path_to_remove, "")
        # Add the missing "module." prefix if missing (DDP prefix)
        if not param_name.startswith("module."):
            param_name = f"module.{param_name}"
        return param_name
