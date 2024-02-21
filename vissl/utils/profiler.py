# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
import logging
import os
from contextlib import contextmanager

from vissl.utils.layer_memory_tracking import null_context


class Profiler:
    """
    Wrapper around the profilers of Pytorch to provide a uniform
    interface, independent of the chosen implementation
    """

    def __init__(self, profiler):
        self.profiler = profiler
        self.started = False
        self.dumped = False

    def __enter__(self):
        self._enter_actions()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.started:
            self._exit_actions(exc_type, exc_val, exc_tb)

    def step(self):
        if self.started:
            self.profiler.step()

    def dump(self, folder: str, rank: int):
        if self.dumped:
            return

        self.export_key_averages(folder, rank)
        self.export_chrome_traces(folder, rank)
        self.dumped = True

    def export_key_averages(self, folder: str, rank: int):
        key_averages = self.profiler.key_averages()
        for stat_type in ["cpu_time", "cuda_time", "cuda_memory_usage"]:
            file_name = f"{stat_type}_rank{rank}.txt"
            file_path = os.path.abspath(os.path.join(folder, file_name))
            logging.info(f"Exporting {stat_type} to: {file_path}")
            with open(file_path, "w") as f:
                print(key_averages.table(sort_by=stat_type), file=f)

    def export_chrome_traces(self, folder: str, rank: int):
        file_name = f"profiler_chrome_trace_rank{rank}.json"
        file_path = os.path.abspath(os.path.join(folder, file_name))
        logging.info(f"Exporting profiling chrome traces to: {file_path}")
        self.profiler.export_chrome_trace(file_path)

    def _enter_actions(self):
        logging.info("Entering profiler...")
        self.profiler.__enter__()
        self.started = True

    def _exit_actions(self, exc_tb, exc_type, exc_val):
        logging.info("Exiting profiler...")
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        self.started = False


class AutoGradProfiler(Profiler):
    """
    Implementation of the Profiler wrapper for the legacy autograd profiler:
    the schedule does not exist in this profiler and must be replaced
    """

    def __init__(self, profiler, wait: int, warmup: int, active: int):
        super().__init__(profiler=profiler)
        self.start_iteration = wait + warmup
        self.end_iteration = self.start_iteration + active
        self.current_iteration = 0
        self.started = False

    def __enter__(self):
        if self.start_iteration == 0:
            self._enter_actions()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.started:
            self._exit_actions(exc_tb, exc_type, exc_val)

    def step(self):
        self.current_iteration += 1
        if self.current_iteration == self.start_iteration:
            self._enter_actions()


@functools.lru_cache(maxsize=1)
def is_nvtx_available():
    try:

        return True
    except ImportError:
        return False


@functools.lru_cache(maxsize=1)
def is_pytorch_profiler_available() -> bool:
    """
    Indicates whether the new pytorch profiler is available
    (available starting from pytorch version 1.8.1)
    """
    try:

        return True
    except ImportError:
        return False


def create_runtime_profiler(
    enabled: bool,
    use_cpu: bool,
    use_cuda: bool,
    wait: int,
    warmup: int,
    active: int,
    legacy_profiler: bool,
):
    """
    Create a runtime profiler with the provided options.

    The type of runtime profiler depends on the pytorch version:
    newer version (above 1.8.1) will use "torch.profiler" instead
    of "torch.autograd.profiler".
    """
    if not enabled:
        return null_context()

    if is_pytorch_profiler_available() and not legacy_profiler:
        import torch.profiler

        profiled_activities = []
        if use_cpu:
            profiled_activities.append(torch.profiler.ProfilerActivity.CPU)
        if use_cuda:
            profiled_activities.append(torch.profiler.ProfilerActivity.CUDA)
        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active)
        profiler = torch.profiler.profile(
            activities=profiled_activities, schedule=schedule
        )
        return Profiler(profiler)
    else:
        import torch.autograd.profiler

        profiler = torch.autograd.profiler.profile(
            enabled=enabled, use_cuda=use_cuda, profile_memory=False
        )
        return AutoGradProfiler(profiler, wait=wait, warmup=warmup, active=active)


@contextmanager
def record_function(name: str, with_tag: str = "##"):
    """
    Context manager to annotate a scope with meta data used for
    profiling. The tag is used to surround the name.
    """
    import torch.autograd.profiler as profiler

    if with_tag:
        name = " ".join([with_tag, name, with_tag])

    if is_nvtx_available():
        import nvtx

        nvtx_context = nvtx.annotate(message=name)
    else:
        nvtx_context = null_context()
    with profiler.record_function(name), nvtx_context:
        yield
