# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
from vissl.data.data_helper import balanced_sub_sampling, unbalanced_sub_sampling


class TestDataLimitSubSampling(unittest.TestCase):
    """
    Testing the DATA_LIMIT underlying sub sampling methods
    """

    def test_unbalanced_sub_sampling(self) -> None:
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0])

        indices1 = unbalanced_sub_sampling(len(labels), num_samples=8, skip_samples=0)
        self.assertEqual(8, len(indices1))
        self.assertEqual(len(indices1), len(set(indices1)), "indices must be unique")

        indices2 = unbalanced_sub_sampling(len(labels), num_samples=8, skip_samples=2)
        self.assertEqual(8, len(indices2))
        self.assertEqual(len(indices2), len(set(indices2)), "indices must be unique")

        self.assertTrue(
            np.array_equal(indices1[2:], indices2[:-2]),
            "skipping samples should slide the window",
        )

    def test_balanced_sub_sampling(self) -> None:
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0])
        unique_labels = set(labels)

        indices1 = balanced_sub_sampling(labels, num_samples=8, skip_samples=0)
        values, counts = np.unique(labels[indices1], return_counts=True)
        self.assertEqual(8, len(indices1))
        self.assertEqual(
            set(values),
            set(unique_labels),
            "at least one of each label should be selected",
        )
        self.assertEqual(2, np.min(counts), "at least two of each label is selected")
        self.assertEqual(2, np.max(counts), "at most two of each label is selected")

        indices2 = balanced_sub_sampling(labels, num_samples=8, skip_samples=4)
        self.assertEqual(8, len(indices2))
        self.assertEqual(
            4,
            len(set(indices1) & set(indices2)),
            "skipping samples should slide the window",
        )
