from typing import NamedTuple, Type, List, Union

import numpy as np
import torch
import torch.nn as nn


class TraceForwardEvent(NamedTuple):
    """
    Complementary trace event collected during the forward pass
    to trace the memory increase and the memory taken by activations
    """
    memory_diff: int
    memory_activations: int


class TraceBackwardEvent(NamedTuple):
    """
    Complementary trace event collected during the forward pass
    to trace the memory taken by activations
    """
    memory_activations: int


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
    def forward_traces(self):
        return (t for t in self.memory_traces if t.is_forward)

    @property
    def backward_traces(self):
        return (t for t in self.memory_traces if not t.is_forward)

    @property
    def max_memory_allocated(self):
        return max(t.allocated for t in self.memory_traces)

    @property
    def max_memory_cached(self):
        return max(t.reserved for t in self.memory_traces)

    @property
    def summary(self):
        total_diff = sum(t.event.memory_diff for t in self.forward_traces)
        total_act = sum(t.event.memory_activations for t in self.forward_traces)
        return {
            'max_memory_allocated': self.max_memory_allocated,
            'max_memory_cached': self.max_memory_cached,
            'total_forward_activation': total_act,
            'total_forward_diff': total_diff,
        }

    def top_activation_producers(self, top: int = 10):
        return sorted(self.forward_traces, key=lambda a: a.memory, reverse=True)[:top]

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
                self.memory_traces.append(LayerMemoryTrace(
                    module_name=name,
                    module_params=self.get_parameter_size(module),
                    allocated=allocated,
                    reserved=reserved,
                    is_forward=True,
                    event=TraceForwardEvent(
                        memory_diff=allocated - self._memory_pre_forward,
                        memory_activations=activations,
                    )
                ))

            # Clean previous forward call values
            self._previous_module_name = None
            self._memory_pre_forward = 0

        return _post_forward_hook

    def _create_backward_hook(self, name: str):
        def _backward_hook(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
            if name not in self._traced_module_names:
                return

            ys = self._filter_allocated_output(grad_input, grad_output)
            memory = sum(self._get_module_output_size(y) for y in ys)
            allocated, reserved = self._capture_memory()
            self.memory_traces.append(LayerMemoryTrace(
                module_name=name,
                module_params=self.get_parameter_size(module),
                allocated=allocated,
                reserved=reserved,
                is_forward=False,
                event=TraceBackwardEvent(memory_activations=memory)
            ))
        return _backward_hook

    @staticmethod
    def _capture_memory():
        torch.cuda.synchronize()
        allocated_mb = torch.cuda.memory_allocated()
        reserved_mb = torch.cuda.memory_reserved()
        return allocated_mb, reserved_mb

    def show_plots(self, figsize=(16, 12), capture: bool = False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=2)
        ax[0, 0].set_title('memory allocated')
        ax[0, 0].plot([trace.allocated for trace in self.memory_traces])
        ax[0, 1].set_title('memory reserved')
        ax[0, 1].plot([trace.reserved for trace in self.memory_traces])
        ax[1, 0].set_title('activation allocations')
        forward_activations = [t.event.memory_activations if t.is_forward else 0 for t in self.memory_traces]
        backward_activations = [t.event.memory_activations if not t.is_forward else 0 for t in self.memory_traces]
        ax[1, 0].plot(forward_activations, label="forward")
        ax[1, 0].plot(backward_activations, label="backward")
        ax[1, 0].legend()
        ax[1, 1].set_title('parameter memory')
        ax[1, 1].plot([a.module_params for a in self.memory_traces])
        if not capture:
            plt.show()
        else:
            return matplotlib_figure_to_image(fig)

    @staticmethod
    def get_parameter_count(module):
        return sum(p.numel() for p in module.parameters())

    @classmethod
    def get_parameter_size(cls, module):
        return sum(p.numel() * cls._get_dtype_size(p) for p in module.parameters())

    @classmethod
    def _get_forward_shapes(cls, xs):
        if isinstance(xs, torch.Tensor):
            return xs.shape
        else:
            return [cls._get_forward_shapes(x) for x in xs]

    @classmethod
    def _get_gradient_shapes(cls, xs):
        if isinstance(xs, torch.Tensor):
            return xs.shape
        elif isinstance(xs, tuple) or isinstance(xs, list):
            return [cls._get_gradient_shapes(x) for x in xs if x is not None]
        return None

    @classmethod
    def _get_module_output_size(cls, xs):
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

    @staticmethod
    def _is_same_storage(x: torch.Tensor, y: torch.Tensor):
        return x.storage().data_ptr() == y.storage().data_ptr()

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

    @classmethod
    def _filter_allocated_output(cls, inputs, outputs):
        xs = cls._collect_tensors(inputs)
        ys = cls._collect_tensors(outputs)
        return [y for y in ys if all(not cls._is_same_storage(x, y) for x in xs)]


def matplotlib_figure_to_image(fig):
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
