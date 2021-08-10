# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


# AP routine of the Holidays and INSTRE package
# Credits: Matthijs Douze
def score_ap_from_ranks_1(ranks, nres):
    """
    Compute the average precision of one search.

    Args:
        ranks: ordered list of ranks of true positives
        nres: total number of positives in dataset

    Returns:
        ap (float): the average precision following the Holidays and the INSTRE package
    """
    # accumulate trapezoids in PR-plot
    ap = 0.0
    # All have an x-size of:
    recall_step = 1.0 / nres
    for ntp, rank in enumerate(ranks):
        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far, rank = nb of retrieved items so far
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)
        # y-size on right side of trapezoid: ntp and rank are increased by one
        precision_1 = (ntp + 1) / float(rank + 1)
        ap += (precision_1 + precision_0) * recall_step / 2.0
    return ap


# Credits: https://github.com/filipradenovic/revisitop/blob/master/python/evaluate.py
def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Args:
        ranks: zero-based ranks of positive images
        nres: number of positive images

    Returns:
        ap (float): average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1.0 / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.0

    return ap


# Credits: https://github.com/filipradenovic/revisitop/blob/master/python/evaluate.py
def compute_map(ranks, gnd, kappas):
    """
    Computes the mAP for a given set of returned results.

    Credits:
        https://github.com/filipradenovic/revisitop/blob/master/python/evaluate.py

    Usage:
      map = compute_map (ranks, gnd)
            computes mean average precsion (map) only

      map, aps, pr, prs = compute_map (ranks, gnd, kappas)
        -> computes mean average precision (map), average precision (aps) for
           each query
        -> computes mean precision at kappas (pr), precision at kappas (prs) for
           each query

     Notes:
     1) ranks starts from 0, ranks.shape = db_size X #queries
     2) The junk results (e.g., the query itself) should be declared in the gnd
        stuct array
     3) If there are no positive images for some query, that query is excluded
        from the evaluation
    """

    map = 0.0
    nq = ranks.shape[-1]  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0
    for i in np.arange(nq):
        qgnd = np.array(gnd[i]["ok"])

        # Remove pos database queries that are not in the prediction.
        # this is only used in DEBUG_MODE where we limit the number of db images.
        qgnd = qgnd[qgnd < ranks.shape[0]]

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float("nan")
            prs[i, :] = float("nan")
            nempty += 1
            continue
            print(f"Skipping: {i}")

        try:
            qgndj = np.array(gnd[i]["junk"])

            # Remove junk database queries that are not in the prediction.
            # this is only used in DEBUG_MODE where we limit the number of db images.
            qgndj = qgndj[qgndj < ranks.shape[0]]

        except Exception:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs
