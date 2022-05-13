# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    convert_to_distributed_tensor,
    convert_to_normal_tensor,
    is_distributed_training_run,
)


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def all_gather_sizes(x: torch.Tensor) -> List[int]:
    """
    Get the first dimension sizes of the the tensor to gather on each
    of the distributed workers
    """
    dist_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    current_device = torch.device("cuda", torch.cuda.current_device())
    sizes = torch.zeros(size=(world_size,), device=current_device, dtype=torch.int64)
    sizes[dist_rank] = x.shape[0]
    torch.distributed.all_reduce(sizes)
    return list(sizes.cpu().numpy())


def all_gather_heterogeneous(sizes: List[int], x: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather a list of heterogeenous tensors shape in the first
    dimension (different batch sizes)
    """
    current_device = torch.device("cuda", torch.cuda.current_device())
    shape = x.shape[1:]
    all_x = [
        torch.zeros(size=(sizes[i], *shape), device=current_device, dtype=x.dtype)
        for i in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(all_x, x)
    return all_x


def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
