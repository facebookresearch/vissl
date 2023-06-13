# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def calculate_ap(rec, prec):
    """
    Computes the AP under the precision recall curve.
    """
    rec, prec = rec.reshape(rec.size, 1), prec.reshape(prec.size, 1)
    z, o = np.zeros((1, 1)), np.ones((1, 1))
    mrec, mpre = np.vstack((z, rec, o)), np.vstack((z, prec, z))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = 0
    for i in indices:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def get_precision_recall(targets, scores, weights=None):
    """
    [P, R, score, ap] = get_precision_recall(targets, scores, weights)

    Args:
        targets: number of occurrences of this class in the ith image
        scores: score for this image
        weights: 0 or 1 whether where 0 means we should ignore the sample

    Returns:
        P, R: precision and recall
        score: score which corresponds to the particular precision and recall
        ap: average precision
    """
    if weights is not None:
        sortweights = weights
    else:
        sortweights = np.ones((targets.shape[0],), dtype=float)
    valid_inds = np.where(sortweights == 1)
    targets = targets[valid_inds]
    scores = scores[valid_inds]

    # binarize targets
    targets = np.array(targets > 0, dtype=np.float32)
    tog = np.hstack(
        (
            targets[:, np.newaxis].astype(np.float64),
            scores[:, np.newaxis].astype(np.float64),
        )
    )
    ind = np.argsort(scores)
    ind = ind[::-1]
    score = np.array([tog[i, 1] for i in ind])
    sortcounts = np.array([tog[i, 0] for i in ind])

    tp = sortcounts
    fp = sortcounts.copy()
    for i in range(sortcounts.shape[0]):
        if sortcounts[i] >= 1:
            fp[i] = 0.0
        elif sortcounts[i] < 1:
            fp[i] = 1.0
    P = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
    numinst = np.sum(targets)
    R = np.cumsum(tp) / numinst
    ap = calculate_ap(R, P)
    return P, R, score, ap
