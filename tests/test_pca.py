# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from vissl.utils.pca import PCA


class TestDataLimitSubSampling(unittest.TestCase):
    """
    Verification that the implementation of the PCA in VISSL
    works with both tensors and numpy arrays and provide the
    right results
    """

    def test_pca_factoisation(self) -> None:
        X = np.array([[0.5, 0.5], [-0.5, -0.5]])
        pca = PCA(n_components=1)
        Y = pca.fit_transform(X)

        expected = np.array([[0.70710678], [-0.70710678]])
        self.assertTrue(np.allclose(expected, Y, atol=1e-6))

        expected_transform = torch.tensor([[0.7071, 0.7071]], dtype=torch.float64)
        self.assertTrue(torch.allclose(expected_transform, pca.DVt))

    def test_pca_on_numpy_arrays(self) -> None:
        X = np.random.normal(loc=0.0, scale=1.0, size=(100, 16))

        pca = PCA(n_components=8)
        Y = pca.fit_transform(X)
        Z = pca.transform(X)

        self.assertIsInstance(Y, np.ndarray)
        self.assertIsInstance(Z, np.ndarray)
        self.assertEqual((100, 8), Y.shape)
        self.assertTrue(np.array_equal(Y, Z))

    def test_pca_transform_torch_tensor(self) -> None:
        X = np.random.normal(loc=0.0, scale=1.0, size=(100, 16))

        pca = PCA(n_components=8)
        pca.fit(X)
        Y = pca.transform(X)
        Z = pca.transform(torch.from_numpy(X))

        self.assertIsInstance(Y, np.ndarray)
        self.assertIsInstance(Z, torch.Tensor)
        self.assertEqual((100, 8), Y.shape)
        self.assertEqual(torch.Size([100, 8]), Z.shape)
        self.assertTrue(np.array_equal(Y, Z.numpy()))

    def test_pca_fit_torch_tensor(self) -> None:
        X = np.random.normal(loc=0.0, scale=1.0, size=(100, 16))
        X = torch.from_numpy(X)

        pca = PCA(n_components=8)
        pca.fit(X)
        Y = pca.transform(X.numpy())
        Z = pca.transform(X)

        self.assertIsInstance(Y, np.ndarray)
        self.assertIsInstance(Z, torch.Tensor)
        self.assertEqual((100, 8), Y.shape)
        self.assertEqual(torch.Size([100, 8]), Z.shape)
        self.assertTrue(np.array_equal(Y, Z.numpy()))
