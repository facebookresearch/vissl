# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from classy_vision.generic.distributed_util import all_reduce_sum


def distributed_sinkhornknopp(
    Q: torch.Tensor,
    num_iter: int,
    hard_assignment: bool,
    world_size: int,
    use_double_prec: bool,
    use_gpu: bool,
) -> torch.Tensor:
    """
    Apply the distributed sinknorn optimization on the scores matrix to
    find the assignments.

        Input shape: (num_prototypes, batch_size)
        Output shape: (batch_size, num_prototypes)
    """
    eps_num_stab = 1e-12
    with torch.no_grad():
        # remove potential infs in Q
        # replace the inf entries with the max of the finite entries in Q
        mask = torch.isinf(Q)
        ind = torch.nonzero(mask)
        if len(ind) > 0:
            for i in ind:
                Q[i[0], i[1]] = 0
            m = torch.max(Q)
            for i in ind:
                Q[i[0], i[1]] = m
        sum_Q = torch.sum(Q, dtype=Q.dtype)
        all_reduce_sum(sum_Q)
        Q /= sum_Q

        k = Q.shape[0]
        n = Q.shape[1]
        N = world_size * Q.shape[1]

        # we follow the u, r, c and Q notations from
        # https://arxiv.org/abs/1911.05371
        r = torch.ones(k) / k
        c = torch.ones(n) / N
        if use_double_prec:
            r, c = r.double(), c.double()

        if use_gpu:
            r = r.cuda(non_blocking=True)
            c = c.cuda(non_blocking=True)

        for _ in range(num_iter):
            u = torch.sum(Q, dim=1, dtype=Q.dtype)
            all_reduce_sum(u)

            # for numerical stability, add a small epsilon value
            # for non-zero Q values.
            if len(torch.nonzero(u == 0)) > 0:
                Q += eps_num_stab
                u = torch.sum(Q, dim=1, dtype=Q.dtype)
                all_reduce_sum(u)
            u = r / u

            # remove potential infs in "u"
            # replace the inf entries with the max of the finite entries in "u"
            mask = torch.isinf(u)
            ind = torch.nonzero(mask)
            if len(ind) > 0:
                for i in ind:
                    u[i[0]] = 0
                m = torch.max(u)
                for i in ind:
                    u[i[0]] = m

            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0, dtype=Q.dtype)).unsqueeze(0)
        Q = (Q / torch.sum(Q, dim=0, keepdim=True, dtype=Q.dtype)).t().float()

        if hard_assignment:
            index_max = torch.max(Q, dim=1)[1]
            Q.zero_()
            Q.scatter_(1, index_max.unsqueeze(1), 1)
        return Q
