# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle
import threading

import numpy as np
from iopath.common.file_io import g_pathmgr
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from vissl.utils.io import load_file, save_file
from vissl.utils.svm_utils.evaluate import get_precision_recall


# Turning it into a class to encapsulate the training and evaluation logic
# together unlike OSS benchmark which has 3 scripts.
class SVMTrainer:
    """
    SVM trainer that takes care of training (using k-fold cross validation),
    and evaluating the SVMs
    """

    def __init__(self, config, layer, output_dir):
        self.config = config
        self.normalize = config["normalize"]
        self.layer = layer
        self.output_dir = self._get_output_dir(output_dir)
        self.costs_list = self._get_costs_list()
        self.train_ap_matrix = None
        self.cls_list = []

    def _get_output_dir(self, cfg_out_dir):
        odir = f"{cfg_out_dir}/{self.layer}"
        g_pathmgr.mkdirs(odir)
        logging.info(f"Output directory for SVM results: {odir}")
        return odir

    def load_input_data(self, data_file, targets_file):
        """
        Given the input data (features) and targets (labels) files, load the
        features of shape N x D and labels of shape (N,)
        """
        assert g_pathmgr.exists(data_file), "Data file not found. Abort!"
        assert g_pathmgr.exists(targets_file), "Targets file not found. Abort!"
        # load the features and the targets
        logging.info("loading features and targets...")
        targets = load_file(targets_file)
        features = np.array(load_file(data_file)).astype(np.float64)
        assert features.shape[0] == targets.shape[0], "Mismatched #images"
        logging.info(f"Loaded features: {features.shape} and targets: {targets.shape}")
        return features, targets

    def _normalize_features(self, features):
        """
        Normalize the features.
        """
        feats_norm = np.linalg.norm(features, axis=1)
        features = features / (feats_norm + 1e-5)[:, np.newaxis]
        return features

    def _get_costs_list(self):
        """
        Costs values for which SVM training is done. We take costs values
        specified in the costs_list input in the config file. Additionally,
        costs specified to be the powers of a base value are also added
        (assuming the base value is > 0).
        """
        costs_list = self.config["costs"]["costs_list"]
        # we append more costs to the output based on power function
        if self.config["costs"]["base"] > 0.0:
            base = self.config["costs"]["base"]
            start_num, end_num = self.config["costs"]["power_range"]
            for num in range(start_num, end_num):
                costs_list.append(base**num)
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
        out_file = f"{self.output_dir}/cls{cls_cost}.pickle"
        ap_matrix_out_file = f"{self.output_dir}/AP_cls{cls_cost}.npy"
        return out_file, ap_matrix_out_file

    def get_best_cost_value(self):
        """
        During the SVM training, we write the cross validation
        AP value for training at each class and cost value
        combination. We load the AP values and for each
        class, determine the cost value that gives the maximum
        AP. We return the chosen cost values for each class as a
        numpy matrix.
        """
        crossval_ap_file = f"{self.output_dir}/crossval_ap.npy"
        chosen_cost_file = f"{self.output_dir}/chosen_cost.npy"
        if g_pathmgr.exists(crossval_ap_file) and g_pathmgr.exists(chosen_cost_file):
            self.chosen_cost = load_file(chosen_cost_file)
            self.train_ap_matrix = load_file(crossval_ap_file)
            return self.chosen_cost
        if self.train_ap_matrix is None:
            num_classes = len(self.cls_list)
            self.train_ap_matrix = np.zeros((num_classes, len(self.costs_list)))
            for cls_num in range(num_classes):
                for cost_idx in range(len(self.costs_list)):
                    cost = self.costs_list[cost_idx]
                    _, ap_out_file = self._get_svm_model_filename(cls_num, cost)
                    self.train_ap_matrix[cls_num][cost_idx] = float(
                        load_file(ap_out_file)[0]
                    )
        argmax_cls = np.argmax(self.train_ap_matrix, axis=1)
        chosen_cost = [self.costs_list[idx] for idx in argmax_cls]
        logging.info(f"chosen_cost: {chosen_cost}")
        save_file(np.array(self.train_ap_matrix), crossval_ap_file)
        save_file(np.array(chosen_cost), chosen_cost_file)
        logging.info(f"saved crossval_ap AP to file: {crossval_ap_file}")
        logging.info(f"saved chosen costs to file: {chosen_cost_file}")
        self.chosen_cost = chosen_cost
        return np.array(chosen_cost)

    def train_cls(self, features, targets, cls_num):
        """
        Train SVM on the input features and targets for a given class.
        The SVMs are trained for all costs values for the given class. We
        also save the cross-validation AP at each cost value for the given
        class.
        """
        logging.info(f"Training cls: {cls_num}")
        for cost_idx in range(len(self.costs_list)):
            cost = self.costs_list[cost_idx]
            out_file, ap_out_file = self._get_svm_model_filename(cls_num, cost)
            if (
                g_pathmgr.exists(out_file)
                and g_pathmgr.exists(ap_out_file)
                and not self.config.force_retrain
            ):
                logging.info(f"SVM model exists: {out_file}")
                logging.info(f"AP file exists: {ap_out_file}")
                continue

            logging.info(f"Training model with the cost: {cost} cls: {cls_num}")
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
            save_file(np.array([ap_scores.mean()]), ap_out_file)
            logging.info(f"Saving SVM model to: {out_file}")
            with g_pathmgr.open(out_file, "wb") as fwrite:
                pickle.dump(clf, fwrite)

    def train(self, features, targets):
        """
        Train SVMs on the given features and targets for all classes and all the
        costs values.
        """
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
        """
        Test the trained SVM models on the test features and targets values.
        We use the cost per class that gives the maximum cross validation AP on
        the training and load the correspond trained SVM model for the cost value
        and the class.

        Log the test ap to stdout and also save the AP in a file.
        """
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
            model = load_file(model_file)
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
        test_ap_filepath = f"{self.output_dir}/test_ap.npy"
        save_file(np.array(ap_matrix), test_ap_filepath)
        logging.info(f"saved test AP to file: {test_ap_filepath}")
