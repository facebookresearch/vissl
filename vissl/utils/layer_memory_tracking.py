# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from vissl.utils.visualize import matplotlib_figure_to_image


class TraceForwardEvent(NamedTuple):
    """
    Complementary trace event collected during the forward pass
    to trace the memory increase and the memory taken by activations
    """

    memory_diff: int
    memory_activations: int

    def to_dict(self):
        return {
            "memory_diff": self.memory_diff,
            "memory_activations": self.memory_activations,
        }


class TraceBackwardEvent(NamedTuple):
    """
    Complementary trace event collected during the forward pass
    to trace the memory taken by activations
    """

    memory_activations: int

    def to_dict(self):
        return {"memory_activations": self.memory_activations}


class LayerMemoryTrace(NamedTuple):
    """
    Trace event providing the current memory usage
    at each point during forward and backward
    """

    module_name: str
    module_params: int
    allocated: int
    reserved: int
    is_forward: bool
    event: Union[TraceForwardEvent, TraceBackwardEvent]

    def to_dict(self):
        return {
            "module_name": self.module_name,
            "module_params": self.module_params,
            "allocated": self.allocated,
            "reserved": self.reserved,
            "is_forward": self.is_forward,
            "event": self.event.to_dict(),
        }


@dataclass
class LayerwiseMemoryTrackerSummary:
    """
    Summary of the memory allocation during forward/backward
    """

    max_memory_allocated: int
    max_memory_cached: int
    total_activation_allocations: int
    total_forward_allocations: int
    top_forward_activation_producers: List[LayerMemoryTrace]


class LayerwiseMemoryTracker:
    """
    Surround a module to get the graph of the memory consumption during
    the forward and backward, layer by layer, with a breakdown of the
    memory used of the activations versus the total memory consumption

    This class requires the model to be on a CUDA device to track the
    memory consumption
    """

    def __init__(self):
        self.memory_traces: List[LayerMemoryTrace] = []
        self._hooks = []
        self._previous_module_name = None
        self._memory_pre_forward = 0
        self._traced_module_names = set()

    def monitor(self, model: nn.Module):
        """
        Install hooks on the model to track its memory usage
        """
        for name, m in model.named_modules():
            h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
            h2 = m.register_forward_hook(self._create_post_forward_hook(name))
            h3 = m.register_backward_hook(self._create_backward_hook(name))
            self._hooks.extend([h1, h2, h3])
        torch.cuda.empty_cache()

    def clear_traces(self):
        """
        Clear all the traces: new traces will be written on a clean slate
        """
        self.memory_traces.clear()

    def stop(self):
        """
        Stop any form of tracking (removes the hooks used to monitor the model)
        """
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._previous_module_name = None
        self._memory_pre_forward = 0

    @property
    def forward_traces(self) -> List[LayerMemoryTrace]:
        """
        Get the part of the traces which corresponds to the forward pass
        """
        return [t for t in self.memory_traces if t.is_forward]

    @property
    def backward_traces(self) -> List[LayerMemoryTrace]:
        """
        Get the part of the traces which corresponds to the backward pass
        """
        return [t for t in self.memory_traces if not t.is_forward]

    @property
    def max_memory_allocated(self) -> int:
        """
        Peak memory allocated during the forward/backward pass
        """
        return max(t.allocated for t in self.memory_traces)

    @property
    def max_memory_cached(self) -> int:
        """
        Peak memory cached during the forward/backward pass
        """
        return max(t.reserved for t in self.memory_traces)

    @property
    def summary(self) -> LayerwiseMemoryTrackerSummary:
        """
        A quick summary of interesting statistics on the memory usage
        during the forward/backward pass
        """
        total_diff = sum(t.event.memory_diff for t in self.forward_traces)
        total_act = sum(t.event.memory_activations for t in self.forward_traces)
        top_act_producers = self.top_forward_activation_producers(top=10)
        return LayerwiseMemoryTrackerSummary(
            max_memory_allocated=self.max_memory_allocated,
            max_memory_cached=self.max_memory_cached,
            total_activation_allocations=total_act,
            total_forward_allocations=total_diff,
            top_forward_activation_producers=top_act_producers,
        )

    def top_forward_activation_producers(self, top: int = 10):
        """
        What are the top activation producers durinb the forward pass
        """
        return sorted(
            self.forward_traces, key=lambda a: a.event.memory_activations, reverse=True
        )[:top]

    def show_plots(self, figsize=(16, 12), capture: bool = False):
        return compare_memory_traces_in_plot(
            {"run": self.memory_traces}, figsize=figsize, capture=capture
        )

    def _create_pre_forward_hook(self, name: str):
        def _pre_forward_hook(module: nn.Module, inputs):
            allocated, reserved = self._capture_memory()
            self._previous_module_name = name
            self._memory_pre_forward = allocated

        return _pre_forward_hook

    def _create_post_forward_hook(self, name: str):
        def _post_forward_hook(module: nn.Module, inputs, outputs):

            # Only if it is a leaf module
            if name == self._previous_module_name:
                allocated, reserved = self._capture_memory()
                self._traced_module_names.add(name)

                # Get the memory allocated for output activations
                ys = self._filter_allocated_output(inputs, outputs)
                activations = sum(self._get_module_output_size(y) for y in ys)

                # Compute the memory diff + memory taken by the activations
                self.memory_traces.append(
                    LayerMemoryTrace(
                        module_name=name,
                        module_params=self._get_parameter_size(module),
                        allocated=allocated,
                        reserved=reserved,
                        is_forward=True,
                        event=TraceForwardEvent(
                            memory_diff=allocated - self._memory_pre_forward,
                            memory_activations=activations,
                        ),
                    )
                )

            # Clean previous forward call values
            self._previous_module_name = None
            self._memory_pre_forward = 0

        return _post_forward_hook

    def _create_backward_hook(self, name: str):
        def _backward_hook(
            module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor
        ):
            if name not in self._traced_module_names:
                return

            ys = self._filter_allocated_output(grad_input, grad_output)
            memory = sum(self._get_module_output_size(y) for y in ys)
            allocated, reserved = self._capture_memory()
            self.memory_traces.append(
                LayerMemoryTrace(
                    module_name=name,
                    module_params=self._get_parameter_size(module),
                    allocated=allocated,
                    reserved=reserved,
                    is_forward=False,
                    event=TraceBackwardEvent(memory_activations=memory),
                )
            )

        return _backward_hook

    @staticmethod
    def _capture_memory():
        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated()
        reserved_mb = torch.cuda.memory_reserved()
        return allocated_mb, reserved_mb

    @classmethod
    def _get_parameter_size(cls, module):
        return sum(p.numel() * cls._get_dtype_size(p) for p in module.parameters())

    @classmethod
    def _get_module_output_size(cls, xs):
        """
        Return the minimum memory requirement to store the tensors
        provided as parameters
        """
        if isinstance(xs, torch.Tensor):
            x = xs
            p = cls._get_dtype_size(x)
            for x in x.shape:
                p *= x
            return p
        elif isinstance(xs, tuple) or isinstance(xs, list):
            return sum(cls._get_module_output_size(x) for x in xs)
        return 0

    @classmethod
    def _get_dtype_size(cls, x: torch.Tensor):
        return 2 if x.dtype == torch.float16 else 4

    @classmethod
    def _filter_allocated_output(cls, inputs, outputs):
        """
        Only return the outputs that are allocated and not views, reshape
        or stride of the inputs
        """
        xs = cls._collect_tensors(inputs)
        ys = cls._collect_tensors(outputs)
        return [y for y in ys if all(not cls._is_same_storage(x, y) for x in xs)]

    @staticmethod
    def _is_same_storage(x: torch.Tensor, y: torch.Tensor):
        """
        Indicate if x and y share the same storage, meaning that one of them
        is a view, reshape or stride of the other or from a common tensor
        """
        return x.storage().data_ptr() == y.storage().data_ptr()

    @staticmethod
    def _collect_tensors(module_io_tensors):
        """
        Extract the tensors out of the provided input or output of a nn.Module
        """
        tensors = []
        to_visit = [module_io_tensors]
        while to_visit:
            x = to_visit.pop()
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, tuple) or isinstance(x, list):
                to_visit.extend(module_io_tensors)
        return tensors


def compare_memory_traces_in_plot(
    memory_traces_by_job: Dict[str, List[LayerMemoryTrace]],
    figsize: Tuple[int, int] = (16, 12),
    capture: bool = False,
):
    """
    Create a plot of the memory allocation over time during the forward/backward
    passes, with a breakdown of the memory used for activation VS parameters
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=2)
    graph_creator = MemoryGraphCreator()

    ax[0, 0].set_title("memory allocated")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.allocated_memory_curve(ax[0, 0], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[0, 0].legend()

    ax[0, 1].set_title("memory reserved")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.reserved_memory_curve(ax[0, 1], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[0, 1].legend()

    ax[1, 0].set_title("activation allocations")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.activation_allocations(ax[1, 0], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[1, 0].legend()

    ax[1, 1].set_title("parameter memory")
    for job_name, memory_traces in memory_traces_by_job.items():
        graph_creator.module_parameters(ax[1, 1], job_name, memory_traces)
    if len(memory_traces_by_job) > 1:
        ax[1, 1].legend()

    if not capture:
        plt.show()
    else:
        return matplotlib_figure_to_image(fig)


class MemoryGraphCreator:
    """
    Helper class to create graphs to display memory
    """

    def __init__(self):
        import matplotlib

        self.font = {
            "family": matplotlib.rcParams["font.family"],
            "weight": "normal",
            "size": 12,
        }

    def allocated_memory_curve(
        self, ax, job_name: str, memory_traces: List[LayerMemoryTrace]
    ):
        allocated_memory = [t.allocated for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(
            memory_traces, allocated_memory
        )
        ax.plot(x, y_forward, x, y_backward, label=job_name)

        max_index = np.argmax(allocated_memory)
        max_trace = memory_traces[max_index]
        max_module = ".".join(
            [n for n in max_trace.module_name.split(".") if not n.startswith("_")]
        )
        max_phase = "fwd" if max_trace.is_forward else "bwd"
        ax.set_ylim([None, max_trace.allocated * 1.1])
        x_text, y_text = max(0, max_index * 0.8), max_trace.allocated * 1.04
        ax.text(x_text, y_text, f"{max_module} ({max_phase})", fontdict=self.font)

    def reserved_memory_curve(
        self, ax, job_name: str, memory_traces: List[LayerMemoryTrace]
    ):
        reserved_memory = [t.reserved for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(
            memory_traces, reserved_memory
        )
        ax.plot(x, y_forward, x, y_backward, label=job_name)

    def activation_allocations(
        self, ax, job_name: str, memory_traces: List[LayerMemoryTrace]
    ):
        event_allocations = [t.event.memory_activations for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(
            memory_traces, event_allocations
        )
        ax.plot(x, y_forward, x, y_backward, label=job_name)

    def module_parameters(
        self, ax, job_name: str, memory_traces: List[LayerMemoryTrace]
    ):
        module_parameters = [t.module_params for t in memory_traces]
        x, y_forward, y_backward = self._split_forward_backward(
            memory_traces, module_parameters
        )
        ax.plot(x, y_forward, x, y_backward, label=job_name)

    @classmethod
    def _split_forward_backward(cls, memory_traces: List[LayerMemoryTrace], values):
        x_values = np.array(list(range(len(memory_traces))))
        mask_forwards, mask_backwards = cls._mask_forward_backward(memory_traces)
        return (
            x_values,
            np.ma.masked_where(mask_forwards, values),
            np.ma.masked_where(mask_backwards, values),
        )

    @classmethod
    def _mask_forward_backward(cls, memory_traces: List[LayerMemoryTrace]):
        mask_forwards = np.array([t.is_forward for t in memory_traces])
        return mask_forwards, ~mask_forwards
