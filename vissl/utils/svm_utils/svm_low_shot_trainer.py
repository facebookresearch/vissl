# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle

import numpy as np
from iopath.common.file_io import g_pathmgr
from sklearn.svm import LinearSVC
from vissl.utils.io import load_file, save_file
from vissl.utils.svm_utils.evaluate import get_precision_recall
from vissl.utils.svm_utils.svm_trainer import SVMTrainer


class SVMLowShotTrainer(SVMTrainer):
    """
    Train the SVM for the low-shot image classification tasks. Currently,
    datasets like VOC07 and Places205 are supported.

    The trained inherits from the SVMTrainer class and takes care of
    training SVM, evaluating, and aggregate the metrics.
    """

    def __init__(self, config, layer, output_dir):
        super().__init__(config, layer, output_dir)
        self._dataset_name = config["low_shot"]["dataset_name"]
        self.cls_list = []
        self.num_classes = None

    def _get_cls_list(self, targets):
        # classes for which SVM testing should be done
        num_classes, cls_list = None, None
        if self._dataset_name == "voc":
            num_classes = targets.shape[1]
            cls_list = range(num_classes)
        elif "places" in self._dataset_name:
            # each image in places205 has a target cls [0, .... ,204]
            cls_list = list(set(targets[:, 0].tolist()))
            num_classes = len(cls_list)
        else:
            logging.info("Dataset not recognized. Abort!")
        self.cls_list = cls_list
        self.num_classes = num_classes
        logging.info(f"Training SVM for classes: {self.cls_list}")
        logging.info(f"Num classes: {num_classes}")
        return cls_list

    def _get_cls_feats_labels(self, cls, features, targets):
        out_feats, out_cls_labels = None, None
        if self._dataset_name == "voc":
            cls_labels = targets[:, cls].astype(dtype=np.int32, copy=True)
            # find the indices for positive/negative imgs. Remove the ignore label.
            out_data_inds = targets[:, cls] != -1
            out_feats = features[out_data_inds]
            out_cls_labels = cls_labels[out_data_inds]
            # label 0 = not present, set it to -1 as svm train target.
            # Make the svm train target labels as -1, 1.
            out_cls_labels[np.where(out_cls_labels == 0)] = -1
        elif "places" in self._dataset_name:
            cls_labels = targets.astype(dtype=np.int32, copy=True)
            # Remove the ignore label.
            out_data_inds = targets[:, 0] != -1
            out_feats = features[out_data_inds]
            out_cls_labels = cls_labels[out_data_inds]

            # for the given class, get the relevant positive/negative images and
            # Make the svm train target labels as -1, 1.
            cls_inds = np.where(out_cls_labels[:, 0] == cls)
            non_cls_inds = out_cls_labels[:, 0] != cls
            out_cls_labels[non_cls_inds] = -1
            out_cls_labels[cls_inds] = 1
            # finally reshape into the format taken by sklearn svm package.
            out_cls_labels = out_cls_labels.reshape(-1)
        else:
            raise Exception("Dataset not recognized")
        return out_feats, out_cls_labels

    def _get_svm_low_shot_model_filename(self, cls_num, cost, suffix):
        # in case of low-shot training, we train for 5 independent samples
        # (sample{}) and vary low-shot amount (k{}). The input data should have
        # sample{}_k{} information that we extract in suffix below.
        cls_cost = str(cls_num) + "_cost" + str(float(cost))
        out_file = f"{self.output_dir}/cls{cls_cost}_{suffix}.pickle"
        return out_file

    def train(self, features, targets, sample_num, low_shot_kvalue):
        """
        Train SVM on the input features and targets for a given low-shot
        k-value and the independent low-shot sample number.

        We save the trained SVM model for each combination:
            cost value, class number, sample number, k-value
        """
        logging.info("Training Low-shot SVM")
        if self.normalize:
            # normalize the features: N x 9216 (example shape)
            features = self._normalize_features(features)

        # get the class lists to train low-shot SVM classifier on
        self.cls_list = self._get_cls_list(targets)
        for cls_idx in range(len(self.cls_list)):
            cls_num = self.cls_list[cls_idx]
            for cost_idx in range(len(self.costs_list)):
                cost = self.costs_list[cost_idx]
                suffix = f"sample{sample_num}_k{low_shot_kvalue}"
                out_file = self._get_svm_low_shot_model_filename(cls_num, cost, suffix)
                if g_pathmgr.exists(out_file) and not self.config.force_retrain:
                    logging.info(f"SVM model exists: {out_file}")
                    continue
                logging.info(f"Training model with the cost: {cost}")
                clf = LinearSVC(
                    C=cost,
                    class_weight={1: 2, -1: 1},
                    intercept_scaling=1.0,
                    verbose=1,
                    penalty=self.config["penalty"],
                    loss=self.config["loss"],
                    tol=0.0001,
                    dual=self.config["dual"],
                    max_iter=self.config["max_iter"],
                )
                train_feats, train_cls_labels = self._get_cls_feats_labels(
                    cls_num, features, targets
                )
                num_positives = len(np.where(train_cls_labels == 1)[0])
                num_negatives = len(np.where(train_cls_labels == -1)[0])
                logging.info(
                    f"cls: {cls_num} has +ve: {num_positives} -ve: {num_negatives} "
                    f"ratio: {float(num_positives) / num_negatives} "
                    f"features: {train_feats.shape} "
                    f"cls_labels: {train_cls_labels.shape}"
                )
                clf.fit(train_feats, train_cls_labels)
                logging.info(f"Saving SVM model to: {out_file}")
                with g_pathmgr.open(out_file, "wb") as fwrite:
                    pickle.dump(clf, fwrite)
        logging.info(f"Done training: sample: {sample_num} k-value: {low_shot_kvalue}")

    def test(self, features, targets, sample_num, low_shot_kvalue):
        """
        Test the SVM for the input test features and targets for the given:
            low-shot k-value, sample number

        We compute the meanAP across all classes for a given cost value.
        We get the output matrix of shape (1, #costs) for the given sample_num and
        k-value and save the matrix. We use this information to aggregate
        later.
        """
        logging.info("Testing SVM")
        # normalize the features: N x 9216 (example shape)
        if self.normalize:
            # normalize the features: N x 9216 (example shape)
            features = self._normalize_features(features)

        sample_ap_matrix = np.zeros((1, len(self.costs_list)))
        suffix = f"sample{sample_num}_k{low_shot_kvalue}"
        for cost_idx in range(len(self.costs_list)):
            cost = self.costs_list[cost_idx]
            local_cost_ap = np.zeros((self.num_classes, 1))
            for cls_num in self.cls_list:
                logging.info(
                    f"Test sample/k_value/cost/cls: "
                    f"{sample_num}/{low_shot_kvalue}/{cost}/{cls_num}"
                )
                model_file = self._get_svm_low_shot_model_filename(
                    cls_num, cost, suffix
                )
                model = load_file(model_file)
                prediction = model.decision_function(features)
                eval_preds, eval_cls_labels = self._get_cls_feats_labels(
                    cls_num, prediction, targets
                )
                P, R, score, ap = get_precision_recall(eval_cls_labels, eval_preds)
                local_cost_ap[cls_num][0] = ap
            mean_cost_ap = np.mean(local_cost_ap, axis=0)
            sample_ap_matrix[0][cost_idx] = mean_cost_ap
        out_k_sample_file = (
            f"{self.output_dir}/test_ap_sample{sample_num}_k{low_shot_kvalue}.npy"
        )
        save_data = sample_ap_matrix.reshape((1, -1))
        save_file(save_data, out_k_sample_file)
        logging.info(
            f"Saved sample test k_idx AP: {out_k_sample_file} {save_data.shape}"
        )

    def _save_stats(self, output_dir, stat, output):
        out_file = f"{output_dir}/test_ap_{stat}.npy"
        logging.info(f"Saving {stat} to: {out_file} {output.shape}")
        save_file(output, out_file)

    def aggregate_stats(self, k_values, sample_inds):
        """
        Aggregate the test AP across all k-values and independent samples.

        For each low-shot k-value, we obtain the mean, max, min, std AP value.
        Steps:
            1. For each k-value, get the min/max/mean/std value across all the
               independent samples. This results in matrices [#k-values x #classes]
            2. Then we aggregate stats across the classes. For the mean stats in
               step 1, for each k-value, we get the class which has maximum mean.
        """
        logging.info(
            f"Aggregating stats for k-values: {k_values} and sample_inds: {sample_inds}"
        )

        output_mean, output_max, output_min, output_std = [], [], [], []
        for k_idx in range(len(k_values)):
            k_low = k_values[k_idx]
            k_val_output = []
            for inds in range(len(sample_inds)):
                sample_idx = sample_inds[inds]
                file_name = f"test_ap_sample{sample_idx}_k{k_low}.npy"
                filepath = f"{self.output_dir}/{file_name}"
                if g_pathmgr.exists(filepath):
                    k_val_output.append(load_file(filepath))
                else:
                    logging.info(f"file does not exist: {filepath}")
            k_val_output = np.concatenate(k_val_output, axis=0)
            k_low_max = np.max(k_val_output, axis=0).reshape(-1, k_val_output.shape[1])
            k_low_min = np.min(k_val_output, axis=0).reshape(-1, k_val_output.shape[1])
            k_low_mean = np.mean(k_val_output, axis=0).reshape(
                -1, k_val_output.shape[1]
            )
            k_low_std = np.std(k_val_output, axis=0).reshape(-1, k_val_output.shape[1])
            output_mean.append(k_low_mean)
            output_min.append(k_low_min)
            output_max.append(k_low_max)
            output_std.append(k_low_std)

        output_mean = np.concatenate(output_mean, axis=0)
        output_min = np.concatenate(output_min, axis=0)
        output_max = np.concatenate(output_max, axis=0)
        output_std = np.concatenate(output_std, axis=0)

        self._save_stats(self.output_dir, "mean", output_mean)
        self._save_stats(self.output_dir, "min", output_min)
        self._save_stats(self.output_dir, "max", output_max)
        self._save_stats(self.output_dir, "std", output_std)

        argmax_cls = np.argmax(output_mean, axis=1)
        argmax_mean, argmax_min, argmax_max, argmax_std = [], [], [], []
        for idx in range(len(argmax_cls)):
            argmax_mean.append(100.0 * output_mean[idx, argmax_cls[idx]])
            argmax_min.append(100.0 * output_min[idx, argmax_cls[idx]])
            argmax_max.append(100.0 * output_max[idx, argmax_cls[idx]])
            argmax_std.append(100.0 * output_std[idx, argmax_cls[idx]])
        output_results = {}
        for idx in range(len(argmax_max)):
            logging.info(
                f"k-value: {k_values[idx]} mean/min/max/std: {round(argmax_mean[idx], 2)} "
                f"/ {round(argmax_min[idx], 2)} / "
                f"{round(argmax_max[idx], 2)} / "
                f"{round(argmax_std[idx], 2)}"
            )
            output_results[f"k={k_values[idx]}"] = {}
            output_results[f"k={k_values[idx]}"]["mean"] = round(argmax_mean[idx], 2)
            output_results[f"k={k_values[idx]}"]["min"] = round(argmax_min[idx], 2)
            output_results[f"k={k_values[idx]}"]["max"] = round(argmax_max[idx], 2)
            output_results[f"k={k_values[idx]}"]["std"] = round(argmax_std[idx], 2)
        logging.info("All done!!")
        return output_results
