# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import NamedTuple, Set

import numpy as np
import torch
import torch.nn as nn


class ActivationStatistics(NamedTuple):
    """
    Information collected on each activation:
    - "name" of the module and "module_type"
    - current "iteration" of training
    - "mean" value of the activation
    - maximum "spread" from the mean
    """

    name: str
    iteration: int
    module_type: str
    mean: float
    spread: float


class ActivationStatisticsObserver(abc.ABC):
    """
    Abstract interface to override to either collect or stream
    statistics as they are produced
    """

    @abc.abstractmethod
    def consume(self, stat: ActivationStatistics):
        pass


class ActivationStatisticsMonitor:
    """
    Watch via hooks the content of the model's activations during training
    computes basic statistics on them, and stream them to an 'observer'.

    This implementation only traces modules which:
    - do not have child modules (ex: nn.Sequential modules are ignored)
    - are in training mode (the goal is to identify divergences)

    Depending on the 'observer' implementation, the results can be
    accumulated or streamed to tensorboard (or any visualisation tool).

    Args:
        observer (ActivationStatisticsObserver):
            the observer to be notified upon each new trace
        log_frequency (int):
            frequency at which the monitoring of activations is done (ex: 1 is every step)
        ignored_modules (Set[str]):
            set of modules for which tracing the activation is disabled
        sample_feature_map (bool):
            If 'true' feature maps will only contribute one feature to the statistics
            (allows to reduce the compute overhead, while still maintaining the
            ability to detect divergence since in convolution nets, all weights
            are typically used for central elements of the feature map)
    """

    def __init__(
        self,
        observer: ActivationStatisticsObserver,
        log_frequency: int,
        ignored_modules: Set[str] = None,
        sample_feature_map: bool = False,
    ):
        self.observer = observer
        self.log_frequency = log_frequency
        self.ignored_modules = ignored_modules
        self.sample_feature_map = sample_feature_map
        if self.ignored_modules is None:
            self.ignored_modules = {"torch.nn.modules.activation.ReLU"}
        self.iteration = -1
        self._hooks = []
        self._previous_module_name = None

    def set_iteration(self, iteration: int):
        self.iteration = iteration

    def monitor(self, model: nn.Module):
        """
        Install hooks on the model to track its activation statistics
        """
        for name, m in model.named_modules():
            if self._get_qualified_type(m) not in self.ignored_modules:
                h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
                h2 = m.register_forward_hook(self._create_post_forward_hook(name))
                self._hooks.extend([h1, h2])

    def stop(self):
        """
        Stop any form of tracking (removes the hooks used to monitor the model)
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.iteration = -1
        self._previous_module_name = None

    def _should_log(self, module: nn.Module):
        return self.iteration % self.log_frequency == 0 and module.training

    def _create_pre_forward_hook(self, name: str):
        def _pre_forward_hook(module: nn.Module, inputs):
            if self._should_log(module):
                self._previous_module_name = name
            else:
                self._previous_module_name = None

        return _pre_forward_hook

    def _create_post_forward_hook(self, name: str):
        def _post_forward_hook(module: nn.Module, inputs, outputs):

            # Eliminate non-leaf modules as well as modules ignored by the forward
            previous_forward_module_name = self._previous_module_name
            self._previous_module_name = None
            if previous_forward_module_name != name:
                return

            # Collect all outputs
            outputs = self._collect_tensors(outputs)
            if len(outputs) == 0:
                return

            # Compute the statistics
            means = []
            spreads = []
            for output in outputs:
                if output.ndim == 4 and self.sample_feature_map:
                    batch_size, channels, height, width = output.shape
                    output = output[:, :, height // 2, width // 2]
                output = output.detach()
                means.append(output.mean().item())
                spreads.append((output - output.mean()).abs().max())

            # Emit the trace
            self.observer.consume(
                ActivationStatistics(
                    name=name,
                    iteration=self.iteration,
                    module_type=self._get_qualified_type(module),
                    mean=float(np.mean(means)),
                    spread=float(np.max(spreads)),
                )
            )

        return _post_forward_hook

    @staticmethod
    def _get_qualified_type(module: nn.Module):
        return type(module).__module__ + "." + type(module).__name__

    @staticmethod
    def _collect_tensors(module_outputs):
        tensors = []
        to_visit = [module_outputs]
        while to_visit:
            x = to_visit.pop()
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, tuple) or isinstance(x, list):
                to_visit.extend(module_outputs)
        return tensors


class ActivationStatisticsAccumulator(ActivationStatisticsObserver):
    """
    Implementation of ActivationStatisticsObserver which collects
    the statistics in a list (especially useful for tests)
    """

    def __init__(self):
        self.stats = []

    def consume(self, stat: ActivationStatistics):
        self.stats.append(stat)
