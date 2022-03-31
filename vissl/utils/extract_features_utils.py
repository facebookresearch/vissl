# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import re
from typing import List, NamedTuple

import numpy as np
from iopath.common.file_io import g_pathmgr
from vissl.utils.io import load_file, makedir


class ExtractedFeaturesShardPaths(NamedTuple):
    """
    The file paths relevant to load a shard of the extract features
    """

    feature_file: str
    targets_file: str
    indices_file: str


class ExtractedFeatures(NamedTuple):
    """
    The file paths relevant to load a shard of the extract features
    """

    features: np.ndarray
    targets: np.ndarray
    indices: np.ndarray

    @property
    def num_samples(self) -> int:
        return self.features.shape[0]


class ExtractedFeaturesLoader:
    """
    Utility class to deal with features extracted with extract_engine

    For multi-gpu feature extraction, each GPU saves features corresponding to its
    share of the data. This class offers an API to abstract away the loading of
    these extracted features.
    """

    @staticmethod
    def get_shard_file_names(
        input_dir: str,
        split: str,
        layer: str,
        sorted: bool = True,
    ) -> List[ExtractedFeaturesShardPaths]:
        """
        Get the list of files needed to load the extracted features
        """

        # List all the files that are containing the features for a given
        # dataset split and a given layer
        feature_regex = re.compile(rf"(.*)_{split}_{layer}_features.npy")
        prefixes = []
        for file_path in g_pathmgr.ls(input_dir):
            match = feature_regex.match(file_path)
            if match is not None:
                prefixes.append(match.group(1))

        # Sort the shards by file name if required: it might be useful
        # if the algorithm that uses the shards is influenced by ordering
        if sorted:
            prefixes.sort()

        # Yield all the files needed to merge the features dumped on
        # the different GPUs
        shard_paths = []
        for prefix in prefixes:
            feat_file = os.path.join(
                input_dir, f"{prefix}_{split}_{layer}_features.npy"
            )
            targets_file = os.path.join(
                input_dir, f"{prefix}_{split}_{layer}_targets.npy"
            )
            indices_file = os.path.join(input_dir, f"{prefix}_{split}_{layer}_inds.npy")
            shard_paths.append(
                ExtractedFeaturesShardPaths(
                    feature_file=feat_file,
                    targets_file=targets_file,
                    indices_file=indices_file,
                )
            )
        return shard_paths

    @classmethod
    def load_feature_shard(
        cls, paths: ExtractedFeaturesShardPaths, verbose=True, allow_pickle=False
    ) -> ExtractedFeatures:
        """
        Load a shard of the extracted features and returns its content:
        features, targets and indices.
        """
        if verbose:
            logging.info(
                f"Loading:\n{paths.feature_file}\n{paths.targets_file}\n{paths.indices_file}"
            )
        return ExtractedFeatures(
            features=load_file(
                paths.feature_file, verbose=verbose, allow_pickle=allow_pickle
            ),
            targets=load_file(
                paths.targets_file, verbose=verbose, allow_pickle=allow_pickle
            ),
            indices=load_file(
                paths.indices_file, verbose=verbose, allow_pickle=allow_pickle
            ),
        )

    @classmethod
    def load_features(
        cls, input_dir: str, split: str, layer: str, flatten_features: bool = False
    ):
        """
        Merge the features across all GPUs to get the features for the full data.

        Args:
            input_dir (str): input path where the features are dumped
            split (str): whether the features are train or test data features
            layer (str): the features correspond to what layer of the model
            flatten_features (bool): whether or not to flatten the features

        Returns:
            output (Dict): contains features, targets, inds as the keys
        """
        logging.info(f"Merging features: {split} {layer}")

        # Reassemble each feature shard (dumped by a given rank)
        output_feats, output_targets = {}, {}
        shard_paths = cls.get_shard_file_names(input_dir, split=split, layer=layer)
        if not shard_paths:
            raise ValueError(f"No features found for {split} {layer}")

        for shard_path in shard_paths:
            shard_content = cls.load_feature_shard(shard_path)
            for idx in range(shard_content.num_samples):
                index = shard_content.indices[idx]
                output_feats[index] = shard_content.features[idx]
                output_targets[index] = shard_content.targets[idx]

        # Sort the entries by sample index
        indices = np.array(sorted(output_targets.keys()))
        features = np.array([output_feats[i] for i in indices])
        targets = np.array([output_targets[i] for i in indices])

        # Return the outputs
        N = len(indices)
        if flatten_features:
            features = features.reshape(N, -1)
        output = {
            "features": features,
            "targets": targets,
            "inds": indices,
        }
        logging.info(f"Features: {output['features'].shape}")
        logging.info(f"Targets: {output['targets'].shape}")
        logging.info(f"Indices: {output['inds'].shape}")
        return output

    @classmethod
    def map_features_to_img_filepath(
        cls, image_paths: List[str], input_dir: str, split: str, layer: str
    ):
        """
        Map the features across all GPUs to the respective filenames.

        Args:
            image_paths (List[str]): list of image paths. Obtained by dataset.get_image_paths()
            input_dir (str): input path where the features are dumped
            split (str): whether the features are train or test data features
            layer (str): the features correspond to what layer of the model
        """
        logging.info(f"Merging features: {split} {layer}")

        output_dir = f"{input_dir}/features_to_image/{split}/{layer}"
        makedir(output_dir)
        logging.info(f"Saving the mapped features to dir: {output_dir} ...")
        shard_paths = cls.get_shard_file_names(input_dir, split=split, layer=layer)
        if not shard_paths:
            raise ValueError(f"No features found for {split} {layer}")
        for shard_path in shard_paths:
            shard_content = cls.load_feature_shard(shard_path)
            for idx in range(shard_content.num_samples):
                img_index = shard_content.indices[idx]
                img_feat = shard_content.features[idx]
                img_filename = os.path.splitext(
                    os.path.basename(image_paths[img_index])
                )[0]
                out_feat_filename = os.path.join(output_dir, img_filename + ".npy")
                with g_pathmgr.open(out_feat_filename, "wb") as fopen:
                    np.save(fopen, np.expand_dims(img_feat, axis=0))

    @classmethod
    def sample_features(
        cls,
        input_dir: str,
        split: str,
        layer: str,
        num_samples: int,
        seed: int,
        flatten_features: bool = False,
    ):
        """
        This function sample N features across all GPUs in an optimized way, using
        reservoir sampling, to avoid loading all features in memory.

        This is especially useful if the number of feature is huge, cannot hold into
        memory, and we need a small number of them to do an estimation (example of
        k-means on a 1B dataset: we can use a few random million samples to compute
        relatively good centroids)

        Args:
            input_dir (str): input path where the features are dumped
            split (str): whether the features are train or test data features
            layer (str): the features correspond to what layer of the model
            num_samples (int): how many features to sample, if negative load everything
            seed (int): the random seed used for sampling
            flatten_features (bool): whether or not to flatten the features

        Returns:
            output (Dict): contains features, targets, inds as the keys
        """

        if num_samples < 0:
            return cls.load_features(
                input_dir=input_dir,
                split=split,
                layer=layer,
                flatten_features=flatten_features,
            )

        features = []
        targets = []
        indices = []

        # Find the shards containing the features to samples
        count = 0
        shard_paths = cls.get_shard_file_names(input_dir, split=split, layer=layer)
        if not shard_paths:
            raise ValueError(f"No features found for {split} {layer}")

        # Use reservoir sampling to sample some features
        rng = np.random.default_rng(seed)
        for shard_path in shard_paths:
            shard_content = cls.load_feature_shard(shard_path)
            for idx in range(shard_content.num_samples):
                count += 1

                # Fill the reservoir of samples
                if len(features) < num_samples:
                    features.append(shard_content.features[idx])
                    targets.append(shard_content.targets[idx])
                    indices.append(shard_content.indices[idx])

                # Else implement reservoir sampling substitution
                else:
                    pos = rng.integers(low=0, high=count)
                    if pos < num_samples:
                        features[pos] = shard_content.features[idx]
                        targets[pos] = shard_content.targets[idx]
                        indices[pos] = shard_content.indices[idx]

        # Cast the output to numpy arrays
        features = np.stack(features)
        targets = np.stack(targets)
        indices = np.stack(indices)

        # Sort the entries by sample index
        sorted_indices = np.argsort(indices)
        indices = indices[sorted_indices]
        features = features[sorted_indices]
        targets = targets[sorted_indices]

        # Return the output
        if flatten_features:
            N = len(indices)
            features = features.reshape(N, -1)
        output = {"features": features, "targets": targets, "inds": indices}
        logging.info(f"Features: {output['features'].shape}")
        logging.info(f"Targets: {output['targets'].shape}")
        logging.info(f"Indices: {output['inds'].shape}")
        return output
