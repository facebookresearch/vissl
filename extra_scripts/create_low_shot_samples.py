# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
This script is used to create the low-shot data for Places svm trainings.
"""
import argparse
import logging
import random

import numpy as np
from fvcore.common.file_io import PathManager
from vissl.utils.io import load_file, save_file


def sample_symbol(input_targets, output_target, symbol, num):
    logging.info(f"Sampling symbol: {symbol} for num: {num}")
    num_classes = input_targets.shape[1]
    for idx in range(num_classes):
        symbol_data = np.where(input_targets[:, idx] == symbol)[0]
        sampled = random.sample(list(symbol_data), num)
        for index in sampled:
            output_target[index, idx] = symbol
    return output_target


def generate_voc07_low_shot_samples(
    targets, k_values, sample_inds, output_path, layername
):
    k_values = [int(val) for val in k_values]
    # the way sample works is: for each independent sample, and a given k value
    # we create a matrix of the same shape as given targets file. We initialize
    # this matrix with -1 (ignore label). We then sample k positive and
    # (num_classes-1) * k negatives.

    num_classes = targets.shape[1]  # N x 20 shape
    for idx in sample_inds:
        for k in k_values:
            logging.info(f"Sampling: {idx} time for k-value: {k}")
            output = np.ones(targets.shape, dtype=np.int32) * -1
            output = sample_symbol(targets, output, 1, k)
            output = sample_symbol(targets, output, 0, (num_classes - 1) * k)
            output_file = f"{output_path}/{layername}_sample{idx}_k{k}.npy"
            logging.info(f"Saving file: {output_file}")
            save_file(output, output_file)
    logging.info("Done!!")


def find_num_positives(input_targets):
    logging.info("Finding max number of positives per class...")
    num_classes = int(max(input_targets) + 1)
    num_pos = []
    for idx in range(num_classes):
        pos = len(np.where(input_targets == idx)[0])
        num_pos.append(pos)
    return num_pos


def sample_places_data(input_images, input_targets, num):
    sample_imgs, sample_lbls = [], []
    num_classes = int(max(input_targets) + 1)
    logging.info(f"Sampling for num: {num}")
    for idx in range(num_classes):
        sample = input_images[np.where(input_targets == idx)]
        subset_imgs = random.sample(list(sample), num)
        sample_lbls.extend([idx] * num)
        sample_imgs.extend(subset_imgs)
    return sample_imgs, sample_lbls


def generate_places_low_shot_samples(
    targets, k_values, sample_inds, output_path, images_data_file
):
    logging.info("Generating low-shot samples for places data...")
    k_values = [int(val) for val in k_values]

    logging.info(f"Loading images data file: {images_data_file}")
    images = load_file(images_data_file)
    # get the maximum and minumum number of positives per class
    num_pos = find_num_positives(targets)
    logging.info(f"min #num_pos: {min(num_pos)}, max #num_pos: {max(num_pos)}")

    # start sampling now. the way sampling works is:
    # for each independent sample, and a given k value,
    # we create an output targets vector of shape same as the input targets.
    # We initialize this matrix with -1 (ignore values). We sample k positive
    # for each given class and set the value in the matrix to the class number.
    # Thus the resulting matrix has (k * num_classes) samples sampled and
    # remaining are ignored.
    for idx in sample_inds:
        for k in k_values:
            if k > min(num_pos):
                logging.info(f"Skip k: {k} min #pos: {min(num_pos)}")
                continue
            logging.info(f"Sampling: {idx} time for k-value: {k}")
            out_lbls = np.ones(targets.shape, dtype=np.int32) * -1
            out_imgs, out_lbls = sample_places_data(images, targets, k)
            out_img_file = f"{output_path}/train_images_sample{idx}_k{k}.npy"
            out_lbls_file = f"{output_path}/train_labels_sample{idx}_k{k}.npy"
            logging.info(f"Saving imgs file: {out_img_file} {len(out_imgs)}")
            logging.info(f"Saving lbls file: {out_lbls_file} {len(out_lbls)}")
            save_file(out_lbls, out_lbls_file)
            save_file(out_imgs, out_img_file)
    logging.info("Done!!")


def generate_low_shot_samples(
    dataset_name,
    targets,
    k_values,
    sample_inds,
    output_path,
    layername,
    images_data_file,
):
    k_values = k_values.split(",")
    if "voc" in dataset_name:
        generate_voc07_low_shot_samples(
            targets, k_values, sample_inds, output_path, layername
        )
    elif "places" in dataset_name:
        generate_places_low_shot_samples(
            targets, k_values, sample_inds, output_path, images_data_file
        )
    else:
        raise RuntimeError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Sample Low-shot data for Places/VOC")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="choose between places | voc. These are valid choices if your dataset is similar",
    )
    parser.add_argument(
        "--layername",
        type=str,
        default=None,
        help="Layer for which low shot is being general. Valid for voc07 only",
    )
    parser.add_argument(
        "--targets_data_file",
        type=str,
        default=None,
        help="Numpy file containing image labels",
    )
    parser.add_argument(
        "--images_data_file",
        type=str,
        default=None,
        help="Numpy file containing images information",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="path where low-shot samples should be saved",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,2,4,8,16,32,64,96",
        help="Low-shot k-values for svm testing.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of independent samples."
    )
    opts = parser.parse_args()

    assert PathManager.exists(opts.targets_data_file), "Target file not found. Abort"
    targets = load_file(opts.targets_data_file)
    sample_ids = list(range(1, 1 + opts.num_samples))

    generate_low_shot_samples(
        opts.dataset_name,
        targets,
        opts.k_values,
        sample_ids,
        opts.output_path,
        opts.layername,
        opts.images_data_file,
    )


if __name__ == "__main__":
    main()
