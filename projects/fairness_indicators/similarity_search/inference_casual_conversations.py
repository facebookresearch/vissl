# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pandas as pd
import torch
from vissl.utils.io import load_file


K_VALUES = [1, 5, 10, 50]


def load_labels_images_features_metadata(input_data):
    # load the image paths
    test_image_paths = load_file(input_data["test_image_paths"])
    print(f"Number of test image_paths: {test_image_paths.shape}\n")
    train_image_paths = load_file(input_data["train_image_paths"])
    print(f"Number of test image_paths: {train_image_paths.shape}\n")

    # load the features
    test_features = load_file(input_data["test_features"])
    print(f"test_features: {test_features.shape}\n")
    train_features = load_file(input_data["train_features"])
    print(f"train_features: {train_features.shape}\n")

    # load the metadata
    test_metadata = load_file(input_data["test_metadata"])
    if isinstance(test_metadata, list):
        print(f"test_metadata: {len(test_metadata)}")
        print(f"test_metadata keys: {test_metadata[0].keys()}")
    else:
        print(f"test_metadata: {list(test_metadata.values())[0].keys()}")
    train_metadata = load_file(input_data["train_metadata"])
    if isinstance(train_metadata, list):
        print(f"train_metadata: {len(train_metadata)}")
        print(f"train_metadata keys: {train_metadata[0].keys()}")
    else:
        print(f"train_metadata: {list(train_metadata.values())[0].keys()}")

    # load the knn_indices
    test_img_indices = load_file(input_data["test_indices"])
    print(f"Loaded test indices: {len(test_img_indices)}")
    train_img_indices = load_file(input_data["train_indices"])
    print(f"Loaded train indices: {len(train_img_indices)}")
    return (
        test_image_paths,
        test_metadata,
        train_image_paths,
        train_metadata,
        test_features,
        test_img_indices,
        train_features,
        train_img_indices,
    )


def get_cc_skintone_name(skintone_num):
    try:
        skintone_num = int(skintone_num)
        if skintone_num in [1, 2, 3]:
            return "lighter"
        if skintone_num in [4, 5, 6]:
            return "darker"
        else:
            return "unknown"
    except Exception:
        return "unknown"


def get_cc_age_bucket(age_num):
    try:
        age_num = int(age_num)
        if age_num < 18:
            return "<18"
        if age_num >= 18 and age_num < 30:
            return "18-30"
        if age_num >= 30 and age_num < 45:
            return "30-45"
        if age_num >= 45 and age_num < 70:
            return "45-70"
        if age_num >= 70:
            return "70+"
    except Exception:
        return "-1"


def get_utk_age_bucket(age_num):
    try:
        age_num = int(age_num)
        if age_num < 13:
            return "< 13"
        if age_num >= 13 and age_num < 30:
            return "13-30"
        if age_num >= 30 and age_num < 45:
            return "30-45"
        if age_num >= 45:
            return "45+"
    except Exception:
        return "-1"


def parse_cc_data(metadata, image_paths, img_indices, img_features):
    cc_map = {}
    for item in metadata:
        path = item["img_path"]
        path = ("_").join(("/").join(path.split("/")[-3:]).split("_")[:-1])
        gender = item["gender"].lower()
        age = get_cc_age_bucket(item["age"])
        skintone = get_cc_skintone_name(item["skin-type"])
        cc_map[path] = {
            "gender": gender,
            "age": age,
            "skintone": skintone,
            "gender_skintone": f"{gender}_{skintone}",
        }
    # now we filter further based on the image paths actually present in the data
    # and we enter the information about the prediction
    filtered_cc_map, output_features = {}, []
    for item in range(len(img_indices)):
        idx = img_indices[item]
        inp_img_path = image_paths[idx]
        inp_img_path = ("_").join(
            ("/").join(inp_img_path.split("/")[-3:]).split("_")[:-1]
        )
        filtered_cc_map[inp_img_path] = cc_map[inp_img_path]
        filtered_cc_map[inp_img_path].update({"feature": img_features[item]})
        output_features.append(img_features[item])
    output_features = np.vstack(output_features)
    print(f"Returning parsed data and features: {output_features.shape}")
    return filtered_cc_map, output_features


def parse_utk_data(metadata, image_paths, img_indices, img_features):
    utk_map = {}
    for key, val in metadata.items():
        path = f"manifold://ssl_framework/tree/datasets/utk_faces/images/{key}"
        gender = val["gender"].lower()
        age = get_utk_age_bucket(val["age"])
        skintone = val["race"]
        utk_map[path] = {
            "gender": gender,
            "age": age,
            "skintone": skintone,
            "gender_skintone": f"{gender}_{skintone}",
        }
    # now we filter further based on the image paths actually present in the data
    # and we enter the information about the prediction
    filtered_utk_map, output_features = {}, []
    for item in range(len(img_indices)):
        idx = img_indices[item]
        inp_img_path = image_paths[idx]
        filtered_utk_map[inp_img_path] = utk_map[inp_img_path]
        filtered_utk_map[inp_img_path].update({"feature": img_features[item]})
        output_features.append(img_features[item])
    output_features = np.vstack(output_features)
    print(f"Returning parsed data and features: {output_features.shape}")
    return filtered_utk_map, output_features


def get_gender_precision_fairness_indicator(
    train_metadata,
    test_metadata,
    train_features,
    test_features,
    apply_l2_norm=False,
    k_values=None,
):
    k_values = k_values or [1, 5, 10, 50]
    if apply_l2_norm:
        print("Applying L2 norm...")
        from torch import nn

        train_features = torch.from_numpy(train_features).float()
        test_features = torch.from_numpy(test_features).float()
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)
        train_features = train_features.numpy()
        test_features = test_features.numpy()

    similarity = test_features.dot(train_features.T)
    print(f"Similarity: {similarity.shape}")

    test_gt_gender_values = [item["gender"] for item in list(test_metadata.values())]
    train_gender_values = [item["gender"] for item in list(train_metadata.values())]
    print(
        f"test/train ground truth gender list: {len(test_gt_gender_values)}, train_gender_values: {len(train_gender_values)}"
    )
    print(
        f"unique gender values -> test: {set(test_gt_gender_values)}, train: {set(train_gender_values)}"
    )

    neighbor_ranks = np.argsort(-similarity, axis=1)
    num_test_images = neighbor_ranks.shape[0]
    # num_train_images = neighbor_ranks.shape[1]
    predicted_gender_matrix = []
    for row_num in range(num_test_images):
        neighbors_gender_values = [
            train_gender_values[idx] for idx in neighbor_ranks[row_num]
        ]
        predicted_gender_matrix.append(neighbors_gender_values)
    predicted_gender_matrix = np.vstack(predicted_gender_matrix)
    print(f"neighbors matrix: {predicted_gender_matrix.shape}")

    (
        output_gender_precision_fairness_indicator,
        output_gender_precision_attributes_count_map,
    ) = (
        {},
        {},
    )
    for k_val in k_values:
        key_str = f"precision@{k_val}"
        (
            output_gender_precision_fairness_indicator[key_str],
            output_gender_precision_attributes_count_map[key_str],
        ) = ({}, {})

        to_predict_attributes = list(list(test_metadata.values())[0].keys())
        to_predict_attributes.remove("feature")
        preds_values = predicted_gender_matrix[:, :k_val]

        # compute overall precision@k per attribute
        output_attributes_precision_map, output_attributes_count_map = {}, {}

        # compute the overall precision@k
        total_correct = 0
        for idx in range(num_test_images):
            total_correct += len(
                np.where(np.array(preds_values[idx]) == test_gt_gender_values[idx])[0]
            )
        output_attributes_precision_map["overall"] = total_correct / (
            k_val * num_test_images
        )
        for attribute in to_predict_attributes:
            attribute_values = [
                item[attribute] for item in list(test_metadata.values())
            ]
            attribute_correct_incorrect = {}
            for idx in range(num_test_images):
                num_correct = len(
                    np.where(np.array(preds_values[idx]) == test_gt_gender_values[idx])[
                        0
                    ]
                )
                if attribute_values[idx] in attribute_correct_incorrect:
                    attribute_correct_incorrect[attribute_values[idx]]["freq"] = (
                        attribute_correct_incorrect[attribute_values[idx]]["freq"] + 1
                    )
                    attribute_correct_incorrect[attribute_values[idx]][
                        "num_correct"
                    ] = (
                        attribute_correct_incorrect[attribute_values[idx]][
                            "num_correct"
                        ]
                        + num_correct
                    )
                else:
                    attribute_correct_incorrect[attribute_values[idx]] = {}
                    attribute_correct_incorrect[attribute_values[idx]]["freq"] = 1
                    attribute_correct_incorrect[attribute_values[idx]][
                        "num_correct"
                    ] = num_correct
            attribute_accuracy_map, attribute_count_map = {}, {}
            for key, val in attribute_correct_incorrect.items():
                attribute_accuracy_map[key] = val["num_correct"] / (val["freq"] * k_val)
                attribute_count_map[key] = val["freq"]
            output_attributes_precision_map[attribute] = attribute_accuracy_map
            output_attributes_count_map[attribute] = attribute_count_map
        output_gender_precision_fairness_indicator[key_str] = (
            output_attributes_precision_map
        )
        output_gender_precision_attributes_count_map[key_str] = (
            output_attributes_count_map
        )
    return (
        output_gender_precision_fairness_indicator,
        output_gender_precision_attributes_count_map,
    )


def convert_and_print_dataframe(output_gender_precision_fairness_indicator, model_name):
    metric_names = list(output_gender_precision_fairness_indicator.keys())
    attributes_list = []
    for key, value in list(
        output_gender_precision_fairness_indicator[metric_names[0]].items()
    ):
        if isinstance(value, dict):
            attributes_list.extend(sorted(value.keys()))
        else:
            attributes_list.append(key)
    attributes_list.remove("-1")

    dataframe = {"model": [], "metric": []}
    for item in attributes_list:
        dataframe[item] = []

    for metric in metric_names:
        dataframe["model"].append(model_name)
        dataframe["metric"].append(metric)
        for key, value in list(
            output_gender_precision_fairness_indicator[metric].items()
        ):
            if isinstance(value, dict):
                for attr_name in list(value.keys()):
                    if attr_name not in dataframe:
                        continue
                    dataframe[attr_name].append(value[attr_name])
            elif key in dataframe:
                dataframe[key].append(value)
    df = pd.DataFrame(data=dataframe)
    print(df.to_markdown())
    return df


def generate_kNN_utk_to_cc_similarity_gender_analysis(
    utk_cc_input_data, apply_l2_norm=True, k_values=None
):
    k_values = k_values or [1, 5, 10, 50]
    print(
        f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ {utk_cc_input_data['model_name']} ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    )
    (
        test_image_paths,
        test_metadata,
        train_image_paths,
        train_metadata,
        test_features,
        test_img_indices,
        train_features,
        train_img_indices,
    ) = load_labels_images_features_metadata(utk_cc_input_data)

    # UTK has gender labels and Race labels , CC has gender labels and skintone scale labels.
    # Only gender is properly aligned for these so we only focus on gender similarity.
    filtered_cc_map, out_test_featues = parse_cc_data(
        test_metadata, test_image_paths, test_img_indices, test_features
    )
    filtered_utk_map, out_train_features = parse_utk_data(
        train_metadata, train_image_paths, train_img_indices, train_features
    )
    (
        output_gender_precision_fairness_indicator,
        output_gender_precision_attributes_count_map,
    ) = get_gender_precision_fairness_indicator(
        filtered_utk_map,
        filtered_cc_map,
        out_train_features,
        out_test_featues,
        apply_l2_norm=apply_l2_norm,
        k_values=k_values,
    )
    df = convert_and_print_dataframe(
        output_gender_precision_fairness_indicator, utk_cc_input_data["model_name"]
    )
    return df


def calculate_metrics(
    model_name,
    test_features_file,
    test_indices_file,
    train_features_file,
    train_indices_file,
):
    my_model_utk_cc_gender_similarity_input_data = {
        "model_name": model_name,
        "test_image_paths": "/path/to/casual_conversations/aligned_cc_mini_test_images.npy",
        "test_metadata": "/path/to/casual_conversations/test_mini_full_metadata.json",
        "train_image_paths": "https://dl.fbaipublicfiles.com/vissl/fairness/similarity_search/aligned_utk_train_images_gender.npy",
        "train_metadata": "https://dl.fbaipublicfiles.com/vissl/fairness/similarity_search/utk_dataset_metadata.json",
        "test_features": test_features_file,
        "test_indices": test_indices_file,
        "train_features": train_features_file,
        "train_indices": train_indices_file,
    }
    _ = generate_kNN_utk_to_cc_similarity_gender_analysis(
        my_model_utk_cc_gender_similarity_input_data,
        apply_l2_norm=True,
        k_values=K_VALUES,
    )


if __name__ == "__main__":
    model_name = "my model name"
    test_features_file = "/path/to/rank0_chunk0_test_heads_features.npy"
    test_indices_file = "/path/to/rank0_chunk0_test_heads_inds.npy"
    train_features_file = "/path/to/rank0_chunk0_train_heads_features.npy"
    train_indices_file = "/path/to/rank0_chunk0_train_heads_inds.npy"

    calculate_metrics(
        model_name,
        test_features_file,
        test_indices_file,
        train_features_file,
        train_indices_file,
    )
