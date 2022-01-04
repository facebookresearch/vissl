# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
import torch.nn as nn
from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from fairscale.nn import FullyShardedDataParallel
from fairscale.nn.misc import FlattenParamsWrapper
from torch.nn import SyncBatchNorm
from vissl.config.attr_dict import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.layer_memory_tracking import LayerwiseMemoryTracker
from vissl.utils.profiler import create_runtime_profiler


class CudaSynchronizeHook(ClassyHook):
    """
    Hook used to call torch.cuda.synchronize() between layer computations,
    slowing the performance down but decreasing the memory usage when using FSDP
    """

    on_loss_and_meter = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_update = ClassyHook._noop

    @staticmethod
    def is_enabled(model_config: AttrDict):
        return model_config.FSDP_CONFIG.FORCE_SYNC_CUDA

    def __init__(self):
        super().__init__()
        self._hooks = []

    def on_start(self, task: "tasks.ClassyTask") -> None:
        model = task.base_model
        for module_name, m in model.named_modules():
            if self._should_sync_module(module_name, m):
                self._hooks.append(m.register_forward_pre_hook(self._pre_forward_hook))
                self._hooks.append(m.register_forward_hook(self._post_forward_hook))
                self._hooks.append(m.register_backward_hook(self._backward_hook))

    def _should_sync_module(self, module_name: str, m: nn.Module) -> bool:
        """
        Only sync the FSDP modules, so that two successive FSDP module
        do not overlap each other, and other modules should run as fast
        as they can.

        We ignore the SyncBN for each of them is wrapped in an FSDP
        module to switch them back to fp32, and not because they contain
        a lot of parameters
        """
        if isinstance(m, FullyShardedDataParallel):
            wrapped = m._fsdp_wrapped_module
            if isinstance(wrapped, SyncBatchNorm):
                return False
            if isinstance(wrapped, FlattenParamsWrapper):
                if isinstance(wrapped._fpw_module, SyncBatchNorm):
                    return False
            return True
        else:
            return False

    def on_exception(self, task: "tasks.ClassyTask") -> None:
        self._cleanup()

    def on_end(self, task: "tasks.ClassyTask") -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @staticmethod
    def _pre_forward_hook(module: nn.Module, inputs):
        torch.cuda.synchronize()

    @staticmethod
    def _post_forward_hook(module: nn.Module, inputs, outputs):
        torch.cuda.synchronize()

    @staticmethod
    def _backward_hook(module: nn.Module, grad_input, grad_output):
        torch.cuda.synchronize()


class ProfilingHook(ClassyHook):
    """
    Hook used to trigger the profiling features of VISSL
    """

    on_loss_and_meter = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop

    @staticmethod
    def is_enabled(profiling_config: AttrDict):
        """
        Returns whether or not the profiler hook should be instantiated:
        it should be enabled if any of the profiling options is on
        """
        with_runtime_profiling = profiling_config.RUNTIME_PROFILING.USE_PROFILER and (
            profiling_config.RUNTIME_PROFILING.PROFILE_CPU
            or profiling_config.RUNTIME_PROFILING.PROFILE_GPU
        )
        return (
            profiling_config.MEMORY_PROFILING.TRACK_BY_LAYER_MEMORY
            or with_runtime_profiling
        )

    def __init__(self, profiling_config: AttrDict):
        super().__init__()
        self.output_folder = profiling_config.OUTPUT_FOLDER
        self.start_iteration = (
            profiling_config.START_ITERATION + profiling_config.WARMUP_ITERATIONS
        )
        self.end_iteration = self.start_iteration + profiling_config.NUM_ITERATIONS
        self.interrupt_training = profiling_config.STOP_TRAINING_AFTER_PROFILING
        self.dist_rank = get_machine_local_and_dist_rank()[1]
        self.is_profiling_rank = self.dist_rank in profiling_config.PROFILED_RANKS
        self.profile_runtime = (
            self.is_profiling_rank and profiling_config.RUNTIME_PROFILING.USE_PROFILER
        )
        self.runtime_profiler = create_runtime_profiler(
            enabled=self.profile_runtime,
            use_cpu=profiling_config.RUNTIME_PROFILING.PROFILE_CPU,
            use_cuda=profiling_config.RUNTIME_PROFILING.PROFILE_GPU,
            wait=profiling_config.START_ITERATION,
            warmup=profiling_config.WARMUP_ITERATIONS,
            active=profiling_config.NUM_ITERATIONS,
            legacy_profiler=profiling_config.RUNTIME_PROFILING.LEGACY_PROFILER,
        )
        self.profile_by_layer_memory = (
            self.is_profiling_rank
            and profiling_config.MEMORY_PROFILING.TRACK_BY_LAYER_MEMORY
        )
        if self.profile_by_layer_memory:
            logging.info(f"Setting up memory tracker for rank {self.dist_rank}...")
            self.layer_memory_tracker = LayerwiseMemoryTracker()

    def on_start(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the start of training.
        """
        if self.profile_by_layer_memory:
            assert (
                task.use_gpu is True
            ), "Profiling memory usage requires training on GPU"
        if self.profile_by_layer_memory and self.start_iteration == 0:
            self.layer_memory_tracker.monitor(task.base_model)
        if self.profile_runtime:
            self.runtime_profiler.__enter__()

    def on_exception(self, task: "tasks.ClassyTask"):
        if self.profile_by_layer_memory:
            iteration = task.local_iteration_num
            self._dump_memory_stats(iteration)
        if self.profile_runtime:
            self._dump_runtime_profiling_results(stop_profiler=True)

    def on_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of training.
        """
        if self.profile_by_layer_memory:
            self.layer_memory_tracker.stop()
        if self.profile_runtime:
            self._dump_runtime_profiling_results(stop_profiler=True)

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Called after parameter update.
        """
        iteration = task.local_iteration_num
        next_iteration = iteration + 1

        if self.profile_by_layer_memory:
            self._memory_tracking(iteration, task)
        if self.profile_runtime:
            self._runtime_tracking(iteration, task)
        if self.interrupt_training and next_iteration >= self.end_iteration + 1:
            self._interrupt_training(iteration)

    def _runtime_tracking(self, iteration: int, task: "tasks.ClassyTask"):
        """
        Handle the runtime profiling logic:
        - enabling / disabling the tracker depending on the iteration
        - dumping the statistics when profiling ends
        """
        next_iteration = iteration + 1
        self.runtime_profiler.step()
        if next_iteration == self.end_iteration:
            self._dump_runtime_profiling_results(stop_profiler=True)

    def _memory_tracking(self, iteration: int, task: "tasks.ClassyTask"):
        """
        Handle the memory tracking logic:
        - enabling / disabling the tracker depending on the iteration
        - dumping the statistics collected in previous iteration
        - preparing the tracker for the next iteration
        """
        next_iteration = iteration + 1

        # Dump memory statistics
        if self.start_iteration <= iteration < self.end_iteration:
            # TODO (prigoyal): figure out how to save when using non-disk backend
            self._dump_memory_stats(iteration)
            self.layer_memory_tracker.clear_traces()

        # Enable / disable the profiling based on the current iteration
        if next_iteration == self.start_iteration:
            self.layer_memory_tracker.monitor(task.base_model)
        if next_iteration == self.end_iteration:
            self.layer_memory_tracker.stop()

    def _interrupt_training(self, iteration: int):
        raise Exception("End of profiling: shutting down...")

    def _dump_memory_stats(self, iteration: int):
        image = self.layer_memory_tracker.show_plots(capture=True)
        image_name = f"memory_rank_{self.dist_rank}_iteration_{iteration}.jpg"
        image.save(os.path.join(self.output_folder, image_name))
        json_name = f"memory_rank_{self.dist_rank}_iteration_{iteration}.json"
        self.layer_memory_tracker.save_traces(
            os.path.join(self.output_folder, json_name)
        )

    def _dump_runtime_profiling_results(self, stop_profiler: bool = False):
        if stop_profiler:
            self.runtime_profiler.__exit__(None, None, None)
        self.runtime_profiler.dump(folder=self.output_folder, rank=self.dist_rank)
