# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from collections import defaultdict, deque
from time import perf_counter
from typing import List, Mapping, Optional, Tuple

import torch
from torch.cuda import Event as CudaEvent


class PerfTimer:
    """
    Very simple timing wrapper, with context manager wrapping.
    Typical usage:

      with PerfTimer('forward_pass', perf_stats):
          model.forward(data)
      # ...
      with PerfTimer('backward_pass', perf_stats):
          model.backward(loss)
      # ...
      print(perf_stats.report_str())

    Note that timer stats accumulate by name, so you can as if resume them
    by re-using the name.

    You can also use it without context manager, i.e. via start() / stop() directly.

    If supplied PerfStats is constructed with use_cuda_events=True (which is default),
    then Cuda events will be added to correctly track time of async execution
    of Cuda kernels:

      with PerfTimer('foobar', perf_stats):
          some_cpu_work()
          schedule_some_cuda_work()

    In example above, the "Host" column will capture elapsed time from the perspective
    of the Python process, and "CudaEvent" column will capture elapsed time between
    scheduling of Cuda work (within the PerfTimer scope) and completion of this work,
    some of which might happen outside the PerfTimer scope.

    If perf_stats is None, using PerfTimer does nothing.
    """

    def __init__(self, timer_name: str, perf_stats: Optional["PerfStats"]):
        self.skip: bool = False
        if perf_stats is None:
            self.skip = True
            return

        self.name: str = timer_name
        self.elapsed: float = 0.0

        self._last_interval: float = 0.0
        self._perf_stats: PerfStats = perf_stats
        self._is_running: bool = False

        if perf_stats.use_cuda_events():
            self._cuda_event_intervals: List[Tuple[CudaEvent, CudaEvent]] = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exception, traceback):
        self.stop()
        if exc_type is None:
            # Only record timer value if with-context finished without error
            self.record()

        return False  # re-raise if there was exception

    def start(self):
        """
        Start the recording if the perfTimer should not be skipped or if the
        recording is not already in progress.
        If using cuda, we record time of cuda events as well.
        """
        if self.skip or self._is_running:
            return

        self._last_interval = 0.0
        self._is_running = True
        self._start_time: float = perf_counter()
        if self._perf_stats.use_cuda_events():
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()

    def stop(self):
        """
        Stop the recording and update the recording interaval, total time elapsed
        from the beginning of perfTimer recording. If using CUDA, we measure time for
        cuda events and append to cuda interval.
        """
        if self.skip or not self._is_running:
            return

        self._last_interval = perf_counter() - self._start_time
        self.elapsed += self._last_interval

        if self._perf_stats.use_cuda_events():
            # Two cuda events will measure real GPU time within PerfTimer scope:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._cuda_event_intervals.append((self._start_event, end_event))

        self._is_running = False

    def record(self):
        """
        Update the timer. We should only do this if the timer is Not skipped and also
        if the timer has already been stopped.
        """
        if self.skip:
            return
        assert not self._is_running
        self._perf_stats.update_with_timer(self)


class PerfMetric:
    """
    Encapsulates numerical tracking of a single metric, with a `.update(value)` API.
    Under-the-hood this can additionally keep track of sums, (exp.) moving averages,
    sum of squares (e.g. for stdev), filtered values, etc.
    """

    # Coefficient for exponential moving average (EMA).
    # Value of 0.1 means last 8 values account for ~50% of weight.
    EMA_FACTOR = 0.1

    def __init__(self):
        self.last_value: Optional[float] = None
        self.smoothed_value: Optional[float] = None

        self.sum_values: float = 0.0
        self.num_updates: int = 0

    def update(self, value: float):
        self.last_value = value
        if self.smoothed_value is None:
            self.smoothed_value = value
        else:
            # TODO (T47970762): correct for initialization bias
            self.smoothed_value = (
                PerfMetric.EMA_FACTOR * value
                + (1.0 - PerfMetric.EMA_FACTOR) * self.smoothed_value
            )

        self.sum_values += value
        self.num_updates += 1

    def get_avg(self):
        """
        Get the mean value of the metrics recorded.
        """
        if self.num_updates == 0:
            return 0.0
        else:
            return self.sum_values / self.num_updates


class PerfStats:
    """
    Accumulate stats (from timers) over many iterations
    """

    MAX_PENDING_TIMERS = 1000

    def __init__(self, use_cuda_events=True):
        self._host_stats: Mapping[str, PerfMetric] = defaultdict(PerfMetric)
        self._cuda_stats: Mapping[str, PerfMetric] = defaultdict(PerfMetric)

        if use_cuda_events:
            if torch.cuda.is_available():
                self._cuda_pending_timers = deque(maxlen=PerfStats.MAX_PENDING_TIMERS)
            else:
                logging.warning("CUDA unavailable: CUDA events are not logged.")
                self._cuda_pending_timers = None
        else:
            self._cuda_pending_timers = None

    def update_with_timer(self, timer: PerfTimer):
        self._host_stats[timer.name].update(timer._last_interval)

        if self.use_cuda_events():
            if len(self._cuda_pending_timers) >= self._cuda_pending_timers.maxlen:
                logging.error(
                    "Too many pending timers. CudaEvent-based stats will be inaccurate!"
                )
            else:
                self._cuda_pending_timers.append(timer)
            self._process_cuda_events()

    def _process_cuda_events(self):
        """
        Service pending timers. Dequeue timers and aggregate Cuda time intervals,
        until the first "pending" timer (i.e. dependent on a not-yet-ready cuda event).
        """
        while len(self._cuda_pending_timers) > 0:
            timer = self._cuda_pending_timers[0]
            elapsed_cuda = 0.0

            for ev_start, ev_end in timer._cuda_event_intervals:
                if not ev_start.query() or not ev_end.query():
                    # Cuda events associated with this timer aren't ready yet,
                    # stop servicing the queue.
                    return
                # Use seconds (instead of ms) for consistency with "host" timers
                elapsed_cuda += ev_start.elapsed_time(ev_end) / 1000.0

            # All time intervals for this timer are now accounted for.
            # Aggregate stats and pop from pending queue.
            self._cuda_stats[timer.name].update(elapsed_cuda)
            self._cuda_pending_timers.popleft()

    def report_str(self):
        """
        Fancy column-aligned human-readable report.
        If using Cuda events, calling this invokes cuda.synchronize(), which is needed
        to capture pending Cuda work in the report.
        """
        if self.use_cuda_events():
            torch.cuda.synchronize()
            self._process_cuda_events()

        name_width = max(len(k) for k in self._host_stats.keys())

        header = ("{:>" + str(name_width + 4) + "s}  {:>7s}    {:>7s}").format(
            "Timer", "Host", "CudaEvent"
        )
        row_fmt = "{:>" + str(name_width + 4) + "s}: {:>7.2f} ms {:>7.2f} ms"

        rows = []
        rows.append(header)
        for name, metric in self._host_stats.items():
            rows.append(
                row_fmt.format(
                    name,
                    metric.get_avg() * 1000.0,
                    self._cuda_stats[name].get_avg() * 1000.0,
                )
            )
        return "\n".join(rows)

    def use_cuda_events(self):
        return torch.cuda.is_available() and self._cuda_pending_timers is not None

    def __str__(self):
        return str((self._host_stats, self._cuda_stats))
