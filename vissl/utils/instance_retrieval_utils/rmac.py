# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


# Credits: https://github.com/facebookresearch/deepcluster/blob/master/eval_retrieval.py    # NOQA
def normalize_L2(a, dim):
    """
    L2 normalize the input tensor along the specified dimension

    Args:
        a(torch.Tensor): the tensor to normalize
        dim (int): along which dimension to L2 normalize

    Returns:
        a (torch.Tensor): L2 normalized tensor
    """
    norms = torch.sqrt(torch.sum(a**2, dim=dim, keepdim=True))
    return a / norms


def get_rmac_region_coordinates(H, W, L):
    """
    Almost verbatim from Tolias et al Matlab implementation.
    Could be heavily pythonized, but really not worth it...
    Desired overlap of neighboring regions
    """
    ovr = 0.4
    # Possible regions for the long dimension
    steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
    w = np.minimum(H, W)

    b = (np.maximum(H, W) - w) / (steps - 1)
    # steps(idx) regions for long dimension. The +1 comes from Matlab
    # 1-indexing...
    idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - ovr)) + 1

    # Region overplus per dimension
    Wd = 0
    Hd = 0
    if H < W:
        Wd = idx
    elif H > W:
        Hd = idx

    regions_xywh = []
    for l in range(1, L + 1):
        wl = np.floor(2 * w / (l + 1))

        if wl == 0:
            # Edge case when w == 0, the height/width will be made 0,
            # resulting in an invalid crop.
            continue

        wl2 = np.floor(wl / 2 - 1)
        # Center coordinates
        if l + Wd - 1 > 0:
            b = (W - wl) / (l + Wd - 1)
        else:
            b = 0
        cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2

        # Center coordinates
        if l + Hd - 1 > 0:
            b = (H - wl) / (l + Hd - 1)
        else:
            b = 0
        cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2

        for i_ in cenH:
            for j_ in cenW:
                regions_xywh.append([j_, i_, wl, wl])

    # Round the regions. Careful with the borders!
    for i in range(len(regions_xywh)):
        for j in range(4):
            regions_xywh[i][j] = int(round(regions_xywh[i][j]))
        if regions_xywh[i][0] + regions_xywh[i][2] > W:
            regions_xywh[i][0] -= (regions_xywh[i][0] + regions_xywh[i][2]) - W
        if regions_xywh[i][1] + regions_xywh[i][3] > H:
            regions_xywh[i][1] -= (regions_xywh[i][1] + regions_xywh[i][3]) - H

    return np.array(regions_xywh)


# Credits: https://github.com/facebookresearch/deepcluster/blob/master/eval_retrieval.py    # NOQA
# Adapted by: Priya Goyal (prigoyal@fb.com)
def get_rmac_descriptors(features, rmac_levels, pca=None, normalize=True):
    """
    RMAC descriptors. Coordinates are retrieved following Tolias et al.
    L2 normalize the descriptors and optionally apply PCA on the descriptors
    if specified by the user. After PCA, aggregate the descriptors (sum) and
    normalize the aggregated descriptor and return.
    """
    nim, nc, xd, yd = features.size()

    rmac_regions = get_rmac_region_coordinates(xd, yd, rmac_levels)
    rmac_regions = rmac_regions.astype(np.int)
    nr = len(rmac_regions)

    rmac_descriptors = []

    for x0, y0, w, h in rmac_regions:
        desc = features[:, :, y0 : y0 + h, x0 : x0 + w]
        desc = torch.max(desc, 2, keepdim=True)[0]
        desc = torch.max(desc, 3, keepdim=True)[0]
        # insert an additional dimension for the cat to work
        rmac_descriptors.append(desc.view(-1, 1, nc))

    rmac_descriptors = torch.cat(rmac_descriptors, 1)

    if normalize:
        # Can optionally skip normalization -- not recommended.
        # the original RMAC paper normalizes.
        rmac_descriptors = normalize_L2(rmac_descriptors, 2)

    if pca is None:
        return rmac_descriptors

    # PCA + whitening
    npca = pca.n_components
    rmac_descriptors = pca.apply(rmac_descriptors.view(nr * nim, nc))

    if normalize:
        rmac_descriptors = normalize_L2(rmac_descriptors, 1)

    rmac_descriptors = rmac_descriptors.view(nim, nr, npca)

    # Sum aggregation and L2-normalization
    rmac_descriptors = torch.sum(rmac_descriptors, 1)
    if normalize:
        rmac_descriptors = normalize_L2(rmac_descriptors, 1)
    return rmac_descriptors
