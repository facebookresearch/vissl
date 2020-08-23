# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging

import numpy as np
import torch
from vissl.utils.io import load_file, save_file


# Credits: https://github.com/facebookresearch/deepcluster/blob/master/eval_retrieval.py    # NOQA
class PCA(object):
    """
    Fits and applies PCA whitening
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
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
        totenergy = d.sum()
        if self.n_components > 0:
            # if we want to retain all components, n_components = -1
            idx = np.argsort(d)[::-1][: self.n_components]
        else:
            idx = np.argsort(d)[::-1]
        d = d[idx]
        V = V[:, idx]

        logging.info("keeping {} % of the energy".format((d.sum() / totenergy * 100.0)))

        D = np.diag(1.0 / np.sqrt(d))
        self.DVt = torch.from_numpy(np.dot(D, V.T))

    def to_cuda(self):
        self.mean = self.mean.cuda()
        self.DVt = self.DVt.cuda()

    def apply(self, X):
        logging.info("Applying PCA...")
        X = X - self.mean
        num = torch.mm(self.DVt, X.transpose(0, 1)).transpose(0, 1)
        return num


def load_pca(pca_out_fname):
    pca = load_file(pca_out_fname)
    return pca


def train_and_save_pca(features, n_pca, pca_out_fname):
    pca = PCA(n_pca)
    pca.fit(features)
    logging.info(f"Saving PCA features to: {pca_out_fname}")
    save_file(pca, pca_out_fname)
    logging.info(f"Saved PCA features to: {pca_out_fname}")
    return pca
