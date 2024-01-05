# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest
from typing import Tuple

import numpy as np
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader
from vissl.utils.io import save_file
from vissl.utils.test_utils import in_temporary_directory


class TestExtractFeatureUtils(unittest.TestCase):
    def prepare_data(
        self, split: str, layer: str, num_shards: int, feat_shape: Tuple[int, int]
    ):
        batch_size, feat_size = feat_shape
        total = batch_size * num_shards

        # Generate a dataset
        indices = np.arange(0, total)
        features = np.random.random(size=(total, feat_size))
        targets = np.random.randint(low=0, high=10, size=(total, 1))

        # Randomly shuffle it
        permutation = np.random.permutation(total)
        permuted_features = features[permutation]
        permuted_targets = targets[permutation]
        permuted_indices = indices[permutation]

        # And save each part in shards
        for i in range(num_shards):
            shard_features = permuted_features[i * batch_size : (i + 1) * batch_size]
            shard_targets = permuted_targets[i * batch_size : (i + 1) * batch_size]
            shard_indices = permuted_indices[i * batch_size : (i + 1) * batch_size]
            save_file(shard_features, f"chunk{i}_{split}_{layer}_features.npy")
            save_file(shard_targets, f"chunk{i}_{split}_{layer}_targets.npy")
            save_file(shard_indices, f"chunk{i}_{split}_{layer}_inds.npy")

        # Return the data used to generate the files
        return indices, features, targets

    def test_get_shard_file_names(self) -> None:
        with in_temporary_directory() as temp_dir:

            # Generate a bunch of split/feature files
            for split in ["train", "test"]:
                for layer in ["heads", "res5"]:
                    self.prepare_data(
                        split=split, layer=layer, num_shards=2, feat_shape=(10, 16)
                    )

            # Check that we only consider the right files
            paths = ExtractedFeaturesLoader.get_shard_file_names(
                input_dir=temp_dir, split="train", layer="heads"
            )
            feature_files = {os.path.split(path.feature_file)[1] for path in paths}
            self.assertEqual(
                feature_files,
                {"chunk0_train_heads_features.npy", "chunk1_train_heads_features.npy"},
            )

    def test_merge_features(self) -> None:
        with in_temporary_directory() as temp_dir:

            # Save the data we need to merge back
            indices, features, targets = self.prepare_data(
                split="train", layer="heads", num_shards=4, feat_shape=(10, 16)
            )

            # Load the data and verify that it is identical
            output = ExtractedFeaturesLoader.load_features(
                input_dir=temp_dir, split="train", layer="heads"
            )
            self.assertEqual(output["features"].shape[0], 40)
            self.assertTrue(np.array_equal(output["inds"], indices))
            self.assertTrue(np.array_equal(output["targets"], targets))
            self.assertTrue(np.allclose(output["features"], features))

            # Sample the all data (no sampling) and check that it is identical
            output = ExtractedFeaturesLoader.sample_features(
                input_dir=temp_dir, split="train", layer="heads", num_samples=-1, seed=0
            )
            self.assertEqual(output["features"].shape[0], 40)
            self.assertTrue(np.array_equal(output["inds"], indices))
            self.assertTrue(np.array_equal(output["targets"], targets))
            self.assertTrue(np.allclose(output["features"], features))

    def test_sample_features(self) -> None:
        with in_temporary_directory() as temp_dir:
            # Save the data we need to sample from
            indices, features, targets = self.prepare_data(
                split="train", layer="heads", num_shards=4, feat_shape=(10, 16)
            )

            # Load the data and verify that it is identical
            output = ExtractedFeaturesLoader.sample_features(
                input_dir=temp_dir, split="train", layer="heads", num_samples=10, seed=0
            )

            # Check that the number of samples is valid
            self.assertEqual(10, len(output["inds"]))

            # Check that the samples are a subset of the original dataset
            self.assertTrue(
                np.array_equal(output["features"], features[output["inds"]])
            )
            self.assertTrue(np.array_equal(output["targets"], targets[output["inds"]]))
