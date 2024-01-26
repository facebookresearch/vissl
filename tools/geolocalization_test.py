# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from typing import Any, List

import pandas as pd
import torch
from hydra.experimental import compose, initialize_config_module
from iopath.common.file_io import g_pathmgr
from vissl.config import AttrDict
from vissl.data.dataset_catalog import get_data_files
from vissl.hooks import default_hook_generator
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.utils.distributed_launcher import launch_distributed
from vissl.utils.env import set_env_vars
from vissl.utils.hydra_config import convert_to_attrdict, is_hydra_available, print_cfg
from vissl.utils.io import load_file, save_file
from vissl.utils.logger import setup_logging, shutdown_logging


PARTITIONIG_MAP = {
    "cells_50_5000": "coarse",
    "cells_50_2000": "middle",
    "cells_50_1000": "fine",
}


# Adapted from
# https://github.com/TIBHannover/GeoEstimation/blob/8dfc2a96741f496587fb598d9627b294058d4c28/classification/s2_utils.py#L20 # NOQA
class Partitioning:
    def __init__(
        self,
        csv_file: str,
        skiprows=2,
        index_col="class_label",
        col_class_label="hex_id",
        col_latitute="latitude_mean",
        col_longitude="longitude_mean",
    ):
        """
        Required information in CSV:
            - class_indexes from 0 to n
            - respective class labels i.e. hexid
            - latitude and longitude
        """
        with g_pathmgr.open(csv_file, "r") as fopen:
            self._df = pd.read_csv(fopen, index_col=index_col, skiprows=skiprows)
        self._df = self._df.sort_index()

        self._nclasses = len(self._df.index)
        self._col_class_label = col_class_label
        self._col_latitude = col_latitute
        self._col_longitude = col_longitude

        # map class label (hexid) to index
        self._label2index = dict(
            zip(self._df[self._col_class_label].tolist(), list(self._df.index))
        )
        self.name = os.path.splitext(os.path.basename(csv_file))[0]
        self.shortname = PARTITIONIG_MAP[self.name]

    def __len__(self):
        return self._nclasses

    def __repr__(self):
        return f"{self.name} short: {self.shortname} n: {self._nclasses}"

    def get_class_label(self, idx):
        return self._df.iloc[idx][self._col_class_label]

    def get_lat_lng(self, idx):
        x = self._df.iloc[idx]
        return float(x[self._col_latitude]), float(x[self._col_longitude])

    def contains(self, class_label):
        if class_label in self._label2index:
            return True
        return False

    def label2index(self, class_label):
        try:
            return self._label2index[class_label]
        except KeyError:
            raise KeyError(f"unknown label {class_label} in {self}")


# Code from:
# https://github.com/TIBHannover/GeoEstimation/blob/8dfc2a96741f496587fb598d9627b294058d4c28/classification/utils_global.py#L66 # NOQA
def vectorized_gc_distance(latitudes, longitudes, latitudes_gt, longitudes_gt):
    R = 6371
    factor_rad = 0.01745329252
    longitudes = factor_rad * longitudes
    longitudes_gt = factor_rad * longitudes_gt
    latitudes = factor_rad * latitudes
    latitudes_gt = factor_rad * latitudes_gt
    delta_long = longitudes_gt - longitudes
    delta_lat = latitudes_gt - latitudes
    subterm0 = torch.sin(delta_lat / 2) ** 2
    subterm1 = torch.cos(latitudes) * torch.cos(latitudes_gt)
    subterm2 = torch.sin(delta_long / 2) ** 2
    subterm1 = subterm1 * subterm2
    a = subterm0 + subterm1
    c = 2 * torch.asin(torch.sqrt(a))
    gcd = R * c
    return gcd


# Code from:
# https://github.com/TIBHannover/GeoEstimation/blob/8dfc2a96741f496587fb598d9627b294058d4c28/classification/utils_global.py#L66 # NOQA
def gcd_threshold_eval(gc_dists, thresholds):
    # calculate accuracy for given gcd thresolds
    results = {}
    for thres in thresholds:
        results[thres] = torch.true_divide(
            torch.sum(gc_dists <= thres), len(gc_dists)
        ).item()
    return results


def geolocalization_test(cfg: AttrDict, layer_name: str = "heads", topk: int = 1):
    output_dir = get_checkpoint_folder(cfg)
    logging.info(f"Output dir: {output_dir} ...")

    ############################################################################
    # Step 1: Load the mapping file and partition it
    # Also load the test images and targets (latitude/longitude)
    # lastly, load the model predictions
    logging.info(
        f"Loading the label partitioning file: {cfg.GEO_LOCALIZATION.TRAIN_LABEL_MAPPING}"
    )
    partitioning = Partitioning(cfg.GEO_LOCALIZATION.TRAIN_LABEL_MAPPING)

    data_files, label_files = get_data_files("TEST", cfg.DATA)
    test_image_paths = load_file(data_files[0])
    target_lat_long = load_file(label_files[0])
    logging.info(
        f"Loaded val image paths: {test_image_paths.shape}, "
        f"ground truth latitude/longitude: {target_lat_long.shape}"
    )

    prediction_image_indices_filepath = f"{output_dir}/rank0_test_{layer_name}_inds.npy"
    predictions_filepath = f"{output_dir}/rank0_test_{layer_name}_predictions.npy"
    predictions = load_file(predictions_filepath)
    predictions_inds = load_file(prediction_image_indices_filepath)
    logging.info(
        f"Loaded predictions: {predictions.shape}, inds: {predictions_inds.shape}"
    )

    ############################################################################
    # Step 2: Convert the predicted classes to latitude/longitude and compute
    # accuracy at different km thresholds.
    gt_latitudes, gt_longitudes, predicted_lats, predicted_longs = [], [], [], []
    output_metadata = {}
    num_images = len(test_image_paths)
    num_images = min(num_images, len(predictions))
    for idx in range(num_images):
        img_index = predictions_inds[idx]
        inp_img_path = test_image_paths[img_index]
        gt_latitude = float(target_lat_long[img_index][0])
        gt_longitude = float(target_lat_long[img_index][1])
        pred_cls = int(predictions[idx][:topk])
        pred_lat, pred_long = partitioning.get_lat_lng(pred_cls)
        output_metadata[inp_img_path] = {
            "target_lat": gt_latitude,
            "target_long": gt_longitude,
            "pred_lat": pred_lat,
            "pred_long": pred_long,
            "pred_cls": pred_cls,
        }
        gt_latitudes.append(gt_latitude)
        gt_longitudes.append(gt_longitude)
        predicted_lats.append(pred_lat)
        predicted_longs.append(pred_long)

    predicted_lats = torch.tensor(predicted_lats, dtype=torch.float)
    predicted_longs = torch.tensor(predicted_longs, dtype=torch.float)
    gt_latitudes = torch.tensor(gt_latitudes, dtype=torch.float)
    gt_longitudes = torch.tensor(gt_longitudes, dtype=torch.float)
    distances = vectorized_gc_distance(
        predicted_lats,
        predicted_longs,
        gt_latitudes,
        gt_longitudes,
    )

    # accuracy for all distances (in km)
    acc_dict = gcd_threshold_eval(
        distances, thresholds=cfg.GEO_LOCALIZATION.ACC_KM_THRESHOLDS
    )
    gcd_dict = {}
    for gcd_thres, acc in acc_dict.items():
        gcd_dict[f"{gcd_thres}"] = round(acc * 100.0, 4)
    logging.info(f"acc dist in percentage: {gcd_dict}")
    save_file(
        output_metadata,
        f"{output_dir}/output_metadata_predictions.json",
        append_to_json=False,
    )
    save_file(
        gcd_dict,
        f"{output_dir}/metrics.json",
        append_to_json=False,
    )
    return output_metadata, acc_dict


def main(args: Namespace, config: AttrDict):
    # setup logging
    setup_logging(__name__)

    # print the coniguration used
    print_cfg(config)

    # setup the environment variables
    set_env_vars(local_rank=0, node_id=0, cfg=config)

    # extract the label predictions on the test set
    launch_distributed(
        config,
        args.node_id,
        engine_name="extract_label_predictions",
        hook_generator=default_hook_generator,
    )

    geolocalization_test(config)

    # close the logging streams including the filehandlers
    shutdown_logging()


def hydra_main(overrides: List[Any]):
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)
    main(args, config)


def invoke_main() -> None:
    overrides = sys.argv[1:]
    assert is_hydra_available(), "Make sure to install hydra"
    hydra_main(overrides=overrides)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
