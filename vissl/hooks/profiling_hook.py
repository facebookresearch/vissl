# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import logging

from classy_vision import tasks
from classy_vision.generic.distributed_util import is_primary
from classy_vision.hooks.classy_hook import ClassyHook

from vissl.utils.layer_memory_tracking import LayerwiseMemoryTracker


class ProfilingHook(ClassyHook):
    """
    Hook used to enable the profiling features of VISSL
    """

    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_phase_end = ClassyHook._noop

    def __init__(
        self,
        output_folder: str,
        start_iteration: int,
        num_iterations: int,
    ):
        super().__init__()
        self.output_folder = output_folder
        self.start_iteration = start_iteration
        self.end_iteration = start_iteration + num_iterations
        self.track_memory = is_primary()  # TODO - actually make that configurable (we might want secondary)
        if self.track_memory:
            logging.info("Setting up memory tracker...")
            self.layer_memory_tracker = LayerwiseMemoryTracker()

    def on_start(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the start of training.
        """
        if self.track_memory and self.start_iteration == 0:
            self.layer_memory_tracker.monitor(task.base_model)

    def on_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of training.
        """
        if self.track_memory:
            self.layer_memory_tracker.stop()

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Called after parameter update.
        """
        iteration = task.iteration
        if self.track_memory:
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
        if iteration >= self.start_iteration:
            image = self.layer_memory_tracker.show_plots(capture=True)
            # TODO - save the raw data as well!
            image.save(f"memory_iteration_{iteration}.jpg")
            self.layer_memory_tracker.clear_traces()

        # Enable / disable the profiling based on the current iteration
        if next_iteration >= self.start_iteration:
            self.layer_memory_tracker.monitor(task.base_model)
        if next_iteration >= self.end_iteration:
            self.layer_memory_tracker.stop()
