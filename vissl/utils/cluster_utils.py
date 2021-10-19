# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from vissl.config import AttrDict
from vissl.data import build_dataset


@dataclass
class ClusterAssignment:
    config: AttrDict
    cluster_assignments: Dict[str, Dict[int, int]]


class ClusterAssignmentLoader:
    """
    Utility class containing functions to load cluster assignments from files
    and visualize them or export them to different formats
    """

    _CONFIG_KEY = "config"
    _ASSIGN_KEY = "cluster_assignments"

    @classmethod
    def save_cluster_assignment(cls, output_dir: str, assignments: ClusterAssignment):
        output_file = os.path.join(output_dir, "cluster_assignments.torch")
        with g_pathmgr.open(output_file, "wb") as f:
            content = {
                cls._CONFIG_KEY: assignments.config,
                cls._ASSIGN_KEY: assignments.cluster_assignments,
            }
            torch.save(content, f)

    @classmethod
    def load_cluster_assigment(cls, file_path: str) -> ClusterAssignment:
        with g_pathmgr.open(file_path, "rb") as f:
            content = torch.load(f)
            return ClusterAssignment(
                config=content[cls._CONFIG_KEY],
                cluster_assignments=content[cls._ASSIGN_KEY],
            )

    @classmethod
    def save_cluster_assignment_as_dataset(
        cls, output_dir: str, assignments: ClusterAssignment
    ):
        """
        Create a 'disk_filelist' dataset out of the cluster assignments:
        - the inputs are the images
        - the labels are the index of the cluster assigned to the image
        """
        os.makedirs(output_dir, exist_ok=True)
        for split in assignments.cluster_assignments.keys():
            dataset = build_dataset(assignments.config, split)
            image_paths = dataset.get_image_paths()
            assert len(image_paths) == 1, "Multi-dataset not supported yet!"
            image_paths = image_paths[0]

            image_labels = []
            for image_id in range(len(image_paths)):
                image_labels.append(assignments.cluster_assignments[split][image_id])

            images_file_path = os.path.join(output_dir, f"{split.lower()}_images.npy")
            labels_file_path = os.path.join(output_dir, f"{split.lower()}_labels.npy")
            np.save(images_file_path, np.array(image_paths))
            np.save(labels_file_path, np.array(image_labels))


class ClusterVisualizer:
    """
    Helper to visualize the clusters assignment of images

    This class is meant to be used in a notebook and will generate
    visualisations for the different clusters
    """

    def __init__(self, assignment: ClusterAssignment, split: str):
        self.config = assignment.config
        cluster_assignments = assignment.cluster_assignments[split]
        self.cluster_to_image = self._to_cluster_to_image_map(cluster_assignments)
        self.dataset = build_dataset(self.config, split)
        self.data_source = self.dataset.data_objs[0]

    def show_cluster(
        self, cluster_id: int, num_col: int = 4, num_row: int = 3, seed: int = 0
    ):
        """
        Display a random subset of image assigned to the cluster in an
        image matrix

        The number of images is num_col * num_row where num_col and num_row
        are the shape of the image matrix
        """

        import matplotlib.pyplot as plt

        image_ids = self.cluster_to_image[cluster_id]
        images = self._select_images(image_ids, num_col * num_row, seed)
        num_images = len(images)
        fig, ax = plt.subplots(
            figsize=(4 * num_col, 4 * num_row), ncols=num_col, nrows=num_row
        )
        for i in range(num_images):
            row, col = divmod(i, num_col)
            ax[row, col].imshow(images[i])
        plt.show()

    def _select_images(self, image_ids: List[int], num_images: int, seed: int):
        if num_images < len(image_ids):
            np.random.seed(seed)
            chosen_image_ids = np.random.choice(
                image_ids, size=num_images, replace=False
            )
            images = [self.data_source[image_id][0] for image_id in chosen_image_ids]
        else:
            images = [self.data_source[image_id][0] for image_id in image_ids]
        return images

    @staticmethod
    def _to_cluster_to_image_map(
        cluster_assignments: Dict[int, int]
    ) -> Dict[int, List[int]]:
        cluster_to_images = {}
        for image_id, cluster_id in cluster_assignments.items():
            cluster_to_images.setdefault(cluster_id, []).append(image_id)
        return cluster_to_images
