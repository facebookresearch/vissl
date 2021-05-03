# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from classy_vision.generic.distributed_util import all_reduce_mean
from torch.autograd import Function


class SyncNormalizeFunction(Function):
    """
    Adapted from: https://github.com/NVIDIA/apex/blob/master/apex/parallel/sync_batchnorm.py

    Normalizes a NxD input over the first dimension and across all processes.
    """

    @staticmethod
    def forward(ctx, input, eps):
        with torch.no_grad():
            local_mean = torch.mean(input, 0)
            local_sqr_mean = torch.pow(input, 2).mean(0)

            # If running on a distributed setting, perform mean reduction of tensors over
            # all processes.
            mean = all_reduce_mean(local_mean)
            sqr_mean = all_reduce_mean(local_sqr_mean)

            # var(x) = E (( x - mean_x ) ** 2)
            #        = 1 / N * sum ( x - mean_x ) ** 2
            #        = 1 / N * sum (x**2) - mean_x**2
            var = sqr_mean - mean.pow(2)

        ctx.save_for_backward(input, mean, var)
        ctx.eps = eps

        return (input - mean) / torch.sqrt(var + eps)

    @staticmethod
    def backward(ctx, grad_output):
        # mini batch mean & var are calculated by forward path.
        # mu = 1./N*np.sum(h, axis = 0)
        # var = 1./N*np.sum((h-mu)**2, axis = 0)
        last_input, mean, var = ctx.saved_tensors

        eps = ctx.eps
        grad_input = None
        num_features = mean.size()[0]

        # calculate grad_input
        if ctx.needs_input_grad[0]:
            # dh = gamma * (var + eps)**(-1. / 2.) * (dy - np.mean(dy, axis=0)
            #     - (h - mu) * (var + eps)**(-1.0) * np.mean(dy * (h - mu), axis=0))
            mean_dy = grad_output.mean(0)
            mean_dy_xmu = (
                (grad_output * (last_input - mean)).view(-1, num_features).mean(0)
            )
            # If running on a distributed setting, perform mean reduction of tensors over
            # all processes.
            mean_dy = all_reduce_mean(mean_dy)
            mean_dy_xmu = all_reduce_mean(mean_dy_xmu)

            grad_input = (
                grad_output - mean_dy - (last_input - mean) / (var + eps) * mean_dy_xmu
            ) / torch.sqrt(var + eps)

        return grad_input, None
