# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Union

import numpy as np
import torch
from vissl.utils.io import load_file, save_file


# Credits: https://github.com/facebookresearch/deepcluster/blob/master/eval_retrieval.py    # NOQA
class PCA:
    """
    Fits and applies PCA whitening
    """

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: Union[np.ndarray, torch.Tensor]):
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        mean = X.mean(axis=0)
        X -= mean
        self.mean = torch.from_numpy(mean).view(1, -1)
        Xcov = np.dot(X.T, X)
        d, V = np.linalg.eigh(Xcov)

        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            logging.info(f"{n_0} / {d.size} singular values are 0")
            d[d < eps] = eps
        if self.n_components > 0:
            # if we want to retain all components, n_components = -1
            idx = np.argsort(d)[::-1][: self.n_components]
        else:
            idx = np.argsort(d)[::-1]
        d = d[idx]
        V = V[:, idx]

        D = np.diag(1.0 / np.sqrt(d))
        self.DVt = torch.from_numpy(np.dot(D, V.T))

    def to_cuda(self):
        self.mean = self.mean.cuda()
        self.DVt = self.DVt.cuda()

    def apply(self, X: Union[np.ndarray, torch.Tensor]):
        input_is_numpy = False
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
            input_is_numpy = True

        X = X - self.mean
        num = torch.mm(self.DVt, X.transpose(0, 1)).transpose(0, 1)
        return num.numpy() if input_is_numpy else num

    def transform(self, X: Union[np.ndarray, torch.Tensor]):
        return self.apply(X)

    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]):
        self.fit(X)
        return self.transform(X)


def load_pca(pca_out_fname):
    pca = load_file(pca_out_fname)
    return pca


def train_and_save_pca(features, n_pca, pca_out_fname):
    logging.info(
        f"Fitting PCA with { n_pca } dimensions to features of shape: {features.shape}"
    )
    pca = PCA(n_pca)
    pca.fit(features)

    if pca_out_fname:
        logging.info(f"Saving PCA features to: {pca_out_fname}")
        save_file(pca, pca_out_fname, verbose=False)
        logging.info(f"Saved PCA features to: {pca_out_fname}")

    return pca
