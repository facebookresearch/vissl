# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os

from classy_vision import tasks
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.config.attr_dict import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.layer_memory_tracking import LayerwiseMemoryTracker


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
        return profiling_config.MEMORY_PROFILING.TRACK_BY_LAYER_MEMORY

    def __init__(self, profiling_config: AttrDict):
        super().__init__()
        self.output_folder = profiling_config.OUTPUT_FOLDER
        self.start_iteration = profiling_config.START_ITERATION
        self.end_iteration = (
            profiling_config.START_ITERATION + profiling_config.NUM_ITERATIONS
        )
        self.dist_rank = get_machine_local_and_dist_rank()[1]
        self.enabled = self.dist_rank in profiling_config.PROFILED_RANKS
        self.profile_memory = (
            self.enabled and profiling_config.MEMORY_PROFILING.TRACK_BY_LAYER_MEMORY
        )
        if self.profile_memory:
            logging.info(f"Setting up memory tracker for rank {self.dist_rank}...")
            self.layer_memory_tracker = LayerwiseMemoryTracker()

    def on_start(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the start of training.
        """
        if self.profile_memory:
            assert (
                task.use_gpu is True
            ), "Profiling memory usage requires training on GPU"
        if self.profile_memory and self.start_iteration == 0:
            self.layer_memory_tracker.monitor(task.base_model)

    def on_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of training.
        """
        if self.profile_memory:
            self.layer_memory_tracker.stop()

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Called after parameter update.
        """
        if self.profile_memory:
            iteration = task.local_iteration_num
            self._memory_tracking(iteration, task)

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
            image = self.layer_memory_tracker.show_plots(capture=True)
            image_name = f"memory_rank_{self.dist_rank}_iteration_{iteration}.jpg"
            image.save(os.path.join(self.output_folder, image_name))
            json_name = f"memory_rank_{self.dist_rank}_iteration_{iteration}.json"
            with open(json_name, "w") as f:
                json_traces = {
                    "traces": [
                        t.to_dict() for t in self.layer_memory_tracker.memory_traces
                    ]
                }
                json.dump(json_traces, f)
            self.layer_memory_tracker.clear_traces()

        # Enable / disable the profiling based on the current iteration
        if next_iteration == self.start_iteration:
            self.layer_memory_tracker.monitor(task.base_model)
        if next_iteration == self.end_iteration:
            self.layer_memory_tracker.stop()
