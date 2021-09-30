# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import numpy as np


class LabelStatistics:
    """
    Useful statistics and visualisations to explore a dataset
    """

    @classmethod
    def label_statistics(cls, labels: List[int]) -> Dict[str, int]:
        counter = {}
        for label in labels:
            counter[label] = counter.get(label, 0) + 1
        counts = list(counter.values())
        return {
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
            "mean": int(np.mean(counts)),
            "median": int(np.median(counts)),
            "std": int(np.std(counts)),
            "percentile_5": int(np.percentile(counts, 5)),
            "percentile_95": int(np.percentile(counts, 95)),
        }

    @classmethod
    def label_histogram(
        cls, labels: List[int], figsize: Tuple[int, int] = (20, 8)
    ) -> None:
        """
        Compute and display some statistics about labels:
        - number of samples associated to each label
        - histogram of the number of samples by label
        """
        import matplotlib.pyplot as plt

        histogram = cls.compute_histogram(labels)
        histogram = sorted(histogram.items())
        xs = [x for x, _ in histogram]
        ys = [y for _, y in histogram]
        plt.figure(figsize=figsize)
        plt.bar(xs, ys)
        plt.show()

    @staticmethod
    def compute_histogram(labels: List[int]) -> Dict[int, int]:
        # How many samples assigned to each label
        counter = {}
        for label in labels:
            counter[label] = counter.get(label, 0) + 1
        counts = list(counter.values())

        # Histogram of number of samples by centroids
        histogram = {}
        for count in counts:
            histogram[count] = histogram.get(count, 0) + 1
        return histogram
