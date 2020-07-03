# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import os
import pickle
import threading

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from vissl.utils.svm_utils.evaluate import get_precision_recall


# Turning it into a class to encapsulate the training and evaluation logic
# together unlike OSS benchmark which has 3 scripts.
class SVMTrainer(object):
    def __init__(self, config, layer):
        self.config = config
        self.normalize = config["normalize"]
        self.layer = layer
        self.output_dir = self._get_output_dir(self.config["OUTPUT_DIR"])
        self.costs_list = self._get_costs_list()
        self.train_ap_matrix = None
        self.cls_list = []

    def _get_output_dir(self, cfg_out_dir):
        odir = os.path.join(os.path.abspath(cfg_out_dir), self.layer)
        if not os.path.exists(odir):
            os.makedirs(odir)
        logging.info(f"Output directory for SVM results: {odir}")
        return odir

    def load_input_data(self, data_file, targets_file):
        assert os.path.exists(data_file), "Data file not found. Abort!"
        assert os.path.exists(targets_file), "Targets file not found. Abort!"
        # load the features and the targets
        logging.info("loading features and targets...")
        targets = np.load(targets_file, encoding="latin1")
        features = np.array(np.load(data_file, encoding="latin1")).astype(np.float64)
        assert features.shape[0] == targets.shape[0], "Mismatched #images"
        logging.info(f"Loaded features: {features.shape} and targets: {targets.shape}")
        return features, targets

    def _normalize_features(self, features):
        feats_norm = np.linalg.norm(features, axis=1)
        features = features / (feats_norm + 1e-5)[:, np.newaxis]
        return features

    def _get_costs_list(self):
        costs_list = self.config["costs"]["costs_list"]
        # we append more costs to the output based on power function
        if self.config["costs"]["base"] > 0.0:
            base = self.config["costs"]["base"]
            start_num, end_num = self.config["costs"]["power_range"]
            for num in range(start_num, end_num):
                costs_list.append(base ** num)
        self.costs_list = costs_list
        logging.info("Training SVM for costs: {}".format(costs_list))
        return costs_list

    def _get_cls_list(self, targets):
        num_classes = targets.shape[1]
        cls_list = range(num_classes)
        if len(self.config["cls_list"]) > 0:
            cls_list = [int(cls_num) for cls_num in self.config["cls_list"]]
        self.cls_list = cls_list
        logging.info("Training SVM for classes: {}".format(self.cls_list))
        return cls_list

    def _get_svm_model_filename(self, cls_num, cost):
        cls_cost = str(cls_num) + "_cost" + str(float(cost))
        out_file = os.path.join(self.output_dir, "cls" + cls_cost + ".pickle")
        ap_matrix_out_file = os.path.join(self.output_dir, "AP_cls" + cls_cost + ".npy")
        return out_file, ap_matrix_out_file

    def get_best_cost_value(self):
        crossval_ap_file = os.path.join(self.output_dir, "crossval_ap.npy")
        chosen_cost_file = os.path.join(self.output_dir, "chosen_cost.npy")
        if os.path.exists(crossval_ap_file) and os.path.exists(chosen_cost_file):
            self.chosen_cost = np.load(chosen_cost_file)
            self.train_ap_matrix = np.load(crossval_ap_file)
            return self.chosen_cost
        if self.train_ap_matrix is None:
            num_classes = len(self.cls_list)
            self.train_ap_matrix = np.zeros((num_classes, len(self.costs_list)))
            for cls_num in range(num_classes):
                for cost_idx in range(len(self.costs_list)):
                    cost = self.costs_list[cost_idx]
                    _, ap_out_file = self._get_svm_model_filename(cls_num, cost)
                    self.train_ap_matrix[cls_num][cost_idx] = float(
                        np.load(ap_out_file, encoding="latin1")[0]
                    )
        argmax_cls = np.argmax(self.train_ap_matrix, axis=1)
        chosen_cost = [self.costs_list[idx] for idx in argmax_cls]
        logging.info("chosen_cost: {}".format(chosen_cost))
        np.save(crossval_ap_file, np.array(self.train_ap_matrix))
        np.save(chosen_cost_file, np.array(chosen_cost))
        logging.info(f"saved crossval_ap AP to file: {crossval_ap_file}")
        logging.info(f"saved chosen costs to file: {chosen_cost_file}")
        self.chosen_cost = chosen_cost
        return np.array(chosen_cost)

    def train_cls(self, features, targets, cls_num):
        logging.info(f"Training cls: {cls_num}")
        for cost_idx in range(len(self.costs_list)):
            cost = self.costs_list[cost_idx]
            out_file, ap_out_file = self._get_svm_model_filename(cls_num, cost)
            if (
                os.path.exists(out_file)
                and os.path.exists(ap_out_file)
                and not self.config.force_retrain
            ):
                logging.info(f"SVM model exists: {out_file}")
                logging.info(f"AP file exists: {ap_out_file}")
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
            cls_labels = targets[:, cls_num].astype(dtype=np.int32, copy=True)
            # meaning of labels in VOC/COCO original loaded target files:
            # label 0 = not present, set it to -1 as svm train target
            # label 1 = present. Make the svm train target labels as -1, 1.
            cls_labels[np.where(cls_labels == 0)] = -1
            num_positives = len(np.where(cls_labels == 1)[0])
            num_negatives = len(cls_labels) - num_positives
            logging.info(
                f"cls: {cls_num} has +ve: {num_positives} -ve: {num_negatives} "
                f"ratio: {float(num_positives) / num_negatives} "
                f"features: {features.shape} cls_labels: {cls_labels.shape}"
            )
            ap_scores = cross_val_score(
                clf,
                features,
                cls_labels,
                cv=self.config["cross_val_folds"],
                scoring="average_precision",
            )
            self.train_ap_matrix[cls_num][cost_idx] = ap_scores.mean()
            clf.fit(features, cls_labels)
            logging.info(
                f"cls: {cls_num} cost: {cost} AP: {ap_scores} "
                f"mean:{ap_scores.mean()}"
            )
            logging.info(f"Saving cls cost AP to: {ap_out_file}")
            np.save(ap_out_file, np.array([ap_scores.mean()]))
            logging.info(f"Saving SVM model to: {out_file}")
            with open(out_file, "wb") as fwrite:
                pickle.dump(clf, fwrite)

    def train(self, features, targets):
        logging.info("Training SVM")
        if self.normalize:
            # normalize the features: N x 9216 (example shape)
            features = self._normalize_features(features)

        # get the class lists to train: whether all or some
        self.cls_list = self._get_cls_list(targets)
        self.train_ap_matrix = np.zeros((len(self.cls_list), len(self.costs_list)))
        threads = []
        for cls_idx in range(len(self.cls_list)):
            cls_num = self.cls_list[cls_idx]
            threads.append(
                threading.Thread(
                    target=self.train_cls, args=(features, targets, cls_num)
                )
            )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test(self, features, targets):
        logging.info("Testing SVM")
        # normalize the features: N x 9216 (example shape)
        if self.normalize:
            # normalize the features: N x 9216 (example shape)
            features = self._normalize_features(features)
        num_classes = targets.shape[1]
        logging.info("Num test classes: {}".format(num_classes))
        # get the chosen cost that maximizes the cross-validation AP per class
        costs_list = self.get_best_cost_value()

        ap_matrix = np.zeros((num_classes, 1))
        for cls_num in range(num_classes):
            cost = costs_list[cls_num]
            logging.info(f"Testing model for cls: {cls_num} cost: {cost}")
            model_file, _ = self._get_svm_model_filename(cls_num, cost)
            with open(model_file, "rb") as fopen:
                model = pickle.load(fopen, encoding="latin1")
            prediction = model.decision_function(features)
            cls_labels = targets[:, cls_num]
            # meaning of labels in VOC/COCO original loaded target files:
            # label 0 = not present, set it to -1 as svm train target
            # label 1 = present. Make the svm train target labels as -1, 1.
            evaluate_data_inds = targets[:, cls_num] != -1
            eval_preds = prediction[evaluate_data_inds]
            eval_cls_labels = cls_labels[evaluate_data_inds]
            eval_cls_labels[np.where(eval_cls_labels == 0)] = -1
            P, R, score, ap = get_precision_recall(eval_cls_labels, eval_preds)
            ap_matrix[cls_num][0] = ap
        logging.info(f"Mean test AP: {np.mean(ap_matrix, axis=0)}")
        test_ap_filepath = os.path.join(self.output_dir, "test_ap.npy")
        np.save(test_ap_filepath, np.array(ap_matrix))
        logging.info(f"saved test AP to file: {test_ap_filepath}")
