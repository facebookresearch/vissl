# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
from math import ceil, floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from IPython.display import display_html
from matplotlib.ticker import FixedLocator
from vissl.utils.io import load_file, save_file


PRED_TOPK = 5
PRED_CONFIDENCE_THRESHOLD = 0.0

COLUMNS_COMPRESSED = [
    "training",
    "architecture",
    "data",
    "data_source",
    "data_size",
    "threshold",
    "midx",
    "model_name",
    "region",
    "household",
    "country",
    "income",
]
COLUMNS = COLUMNS_COMPRESSED + ["image", "label", "hit", "cover"]

INCOME_BUCKETS = {1.0: "low", 2.0: "medium", 3.0: "high"}

ARCH_NAMES = {
    "rn50": "ResNet50",
    "rg16": "RegNet16",
    "rg128": "RegNet128",
    "rg256": "RegNet256",
    "regnet10B": "RegNet10B",
}
TRAINING_NAMES = {
    "sup": "Supervised",
    "seer": "SEER",
    "swav": "SWaV",
    "uru10x10": "URU",
}
SIZE_NAMES = {"1m": "1M", "32m": "32M", "1b": "1B", "10b": "10B"}

THRESHOLDS = [0.0]
_MODELS = {}

_region_type = pd.CategoricalDtype(
    categories=["Africa", "Asia", "The Americas", "Europe"], ordered=True
)
_model_type = pd.CategoricalDtype(
    categories=["ResNet50", "RegNet16", "RegNet128", "RegNet256", "RegNet10B"],
    ordered=True,
)
_training_type = pd.CategoricalDtype(
    categories=["Supervised", "SWaV", "URU", "SEER"], ordered=True
)
_size_type = pd.CategoricalDtype(categories=["1M", "32M", "1B", "10B"], ordered=True)
_income_type = pd.CategoricalDtype(categories=["low", "medium", "high"], ordered=True)


def load_labels_metadata_predictions_images(input_data, confidence_threshold=0.0):
    # load the image paths
    image_paths = load_file(input_data["image_paths"])
    print(f"Number of image_paths: {image_paths.shape}\n")

    # load the predictions
    predictions = load_file(input_data["predictions"])
    print(f"Number of predictions: {predictions.shape}\n")

    # load the indices
    indices = load_file(input_data["pred_img_indices"])
    print(f"Number of indices: {indices.shape}\n")

    # load the targets
    targets = load_file(input_data["targets"])
    print(f"Number of targets: {targets.shape}\n")

    # load the metadata
    metadata = load_file(input_data["metadata"])
    if isinstance(metadata, list):
        print(f"metadata: {len(metadata)}")
        print(f"metadata keys: {metadata[0].keys()}")
    else:
        print(f"metadata: {list(metadata.values())[0].keys()}")

    # load the label id map
    id_to_label = load_file(input_data["id_to_label_map"])
    print(f"Loaded label_to_id and generated id_to_label map: {len(id_to_label)}")

    # load the confidence scores if provided
    filtered_confidence_scores = []
    if "pred_confidence_scores" in input_data:
        confidence_scores = load_file(input_data["pred_confidence_scores"])
        filtered_confidence_scores, out_predictions = [], []
        for img_idx in range(len(predictions)):
            img_predictions, img_scores = [], []
            for pred_idx in predictions[img_idx]:
                if confidence_scores[img_idx][pred_idx] >= confidence_threshold:
                    img_predictions.append(pred_idx)
                    img_scores.append(
                        str(round(confidence_scores[img_idx][pred_idx], 5))
                    )
            filtered_confidence_scores.append(img_scores)
            out_predictions.append(img_predictions)
        predictions = out_predictions
        print(f"Confidence scores: {len(filtered_confidence_scores)}\n")

    return (
        image_paths,
        predictions,
        indices,
        targets,
        metadata,
        id_to_label,
        filtered_confidence_scores,
    )


def generate_dollar_street_map(
    image_paths, metadata, predictions, id_to_label, confidence_scores=None, topk=5
):
    confidence_scores = confidence_scores or []
    output_ds_metadata = {}
    for entry in metadata:
        img_id = entry["_id"]
        label = entry["label"]
        path = f"{label}/{img_id}"
        output_ds_metadata[path] = {
            "country": entry["country"],
            "concept": entry["label"],
            "country_concept": f"{entry['country']}_{entry['label']}",
            "region": entry["region"],
            "income": entry["income"],
            "url": entry["url"],
        }

    filtered_ds_metadata = {}
    has_confidence_scores = False
    if len(confidence_scores) > 0:
        has_confidence_scores = True
    for idx in range(len(image_paths)):
        pred = predictions[idx][:topk]
        img_predictions = []
        raw_predictions = []
        for item in pred:
            raw_predictions.append(id_to_label[str(item)])
        for item in pred:
            img_predictions.extend(id_to_label[str(item)])
        img_id = os.path.splitext(image_paths[idx].split("/")[-1])[0]
        target_label = image_paths[idx].split("/")[-2]
        path = f"{target_label}/{img_id}"
        if path in output_ds_metadata:
            filtered_ds_metadata[path] = output_ds_metadata[path]
            filtered_ds_metadata[path].update({"prediction": img_predictions})
            filtered_ds_metadata[path].update({"raw_prediction": raw_predictions})
            if has_confidence_scores:
                filtered_ds_metadata[path].update(
                    {"confidence_scores": confidence_scores[idx][:topk]}
                )
        else:
            print(f"Not found: {path}")
    print(f"Output data entries: {len(list(filtered_ds_metadata.keys()))}")
    return filtered_ds_metadata


def get_per_attribute_accuracy(filtered_input_map, predict_attribute="concept"):
    to_predict_attributes = list(list(filtered_input_map.values())[0].keys())
    to_predict_attributes.remove("prediction")
    to_predict_attributes.remove("income")
    to_predict_attributes.remove("url")
    to_predict_attributes.remove("raw_prediction")
    if "confidence_scores" in to_predict_attributes:
        to_predict_attributes.remove("confidence_scores")
    print(to_predict_attributes)
    gt_values = [item[predict_attribute] for item in list(filtered_input_map.values())]
    preds_values = [item["prediction"] for item in list(filtered_input_map.values())]

    # we compute the overall accuracy
    correct, total = 0, 0
    for idx in range(len(gt_values)):
        total += 1
        correct += int(gt_values[idx] in list(set(preds_values[idx])))
    overall_accuracy = round(100.0 * (correct / total), 3)
    print(f"Overall accuracy: {overall_accuracy}")

    output_attributes_acc_map = {}
    for attribute in to_predict_attributes:
        attribute_values = [
            item[attribute] for item in list(filtered_input_map.values())
        ]

        attribute_correct_incorrect = {}
        for idx in range(len(gt_values)):
            is_correct = int(gt_values[idx] in list(set(preds_values[idx])))
            if attribute_values[idx] in attribute_correct_incorrect:
                attribute_correct_incorrect[attribute_values[idx]]["freq"] = (
                    attribute_correct_incorrect[attribute_values[idx]]["freq"] + 1
                )
                attribute_correct_incorrect[attribute_values[idx]]["correct"] = (
                    attribute_correct_incorrect[attribute_values[idx]]["correct"]
                    + is_correct
                )
            else:
                attribute_correct_incorrect[attribute_values[idx]] = {}
                attribute_correct_incorrect[attribute_values[idx]]["freq"] = 1
                attribute_correct_incorrect[attribute_values[idx]][
                    "correct"
                ] = is_correct
        attribute_accuracy_map = {}
        for key, val in attribute_correct_incorrect.items():
            attribute_accuracy_map[key] = {}
            attribute_accuracy_map[key]["hit_rate"] = round(
                (val["correct"] / val["freq"]) * 100.0, 3
            )
            attribute_accuracy_map[key]["miss_rate"] = round(
                100.0 - attribute_accuracy_map[key]["hit_rate"], 3
            )
            attribute_accuracy_map[key]["total"] = val["freq"]
        output_attributes_acc_map[attribute] = attribute_accuracy_map
    return output_attributes_acc_map


def create_and_print_dataframe(output_attributes_acc_map):
    countries_names = list(output_attributes_acc_map["country"].keys())
    concept_names = list(output_attributes_acc_map["concept"].keys())
    print(f"Total : concepts={len(concept_names)}, countries={len(countries_names)}")

    COUNTRY_OFFSET = 0
    NUM_COUNTRIES = 12
    while COUNTRY_OFFSET <= len(countries_names):
        d = {"label": [], "metric": []}
        for country in countries_names[
            COUNTRY_OFFSET : min((COUNTRY_OFFSET + NUM_COUNTRIES), len(countries_names))
        ]:
            d[country] = []

        for concept in concept_names:
            d["label"].append(concept)
            d["label"].append("")
            d["metric"].append("TPR (hit_rate)")
            d["metric"].append("#images")
            for country in countries_names[
                COUNTRY_OFFSET : min(
                    (COUNTRY_OFFSET + NUM_COUNTRIES), len(countries_names)
                )
            ]:
                country_concept = f"{country}_{concept}"
                if country_concept in output_attributes_acc_map["country_concept"]:
                    d[country].append(
                        float(
                            output_attributes_acc_map["country_concept"][
                                country_concept
                            ]["hit_rate"]
                        )
                    )
                    d[country].append(
                        float(
                            output_attributes_acc_map["country_concept"][
                                country_concept
                            ]["total"]
                        )
                    )
                else:
                    d[country].append("")
                    d[country].append("")
        df = pd.DataFrame(data=d)
        print(df.to_markdown())
        print("\n")
        COUNTRY_OFFSET += NUM_COUNTRIES


def generate_dollar_street_analysis(input_data, topk=5, confidence_threshold=0.0):
    print(
        f"================= {input_data['model_name']}, t={confidence_threshold}, top-{topk} ====================="
    )
    (
        image_paths,
        predictions,
        indices,
        targets,
        metadata,
        id_to_label,
        confidence_scores,
    ) = load_labels_metadata_predictions_images(
        input_data, confidence_threshold=confidence_threshold
    )
    output_metadata_map = generate_dollar_street_map(
        image_paths, metadata, predictions, id_to_label, confidence_scores, topk=topk
    )
    output_attributes_acc_map = get_per_attribute_accuracy(output_metadata_map)
    print(f"t={confidence_threshold}, top-{topk}")
    # create_and_print_dataframe(output_attributes_acc_map)
    return output_attributes_acc_map, output_metadata_map


def parse_file_name(model_path):
    """
    Customize this to work for your models
    """
    fields = Path(model_path).name.split("_")
    training, architecture = fields[:2]
    if training == "supervised":
        training = "sup"
    if "regnet" in architecture:
        if "256" in architecture:
            architecture = "rg256"
        elif "128" in architecture:
            architecture = "rg128"
        elif "16" in architecture:
            architecture = "rg16"
    for i in range(2, len(fields)):
        if fields[i].startswith("in"):
            tdata, tdata_source, tdata_size = fields[i], "in", "1m"
            break
        elif fields[i].startswith("ig"):
            tdata, tdata_source, tdata_size = fields[i], "ig", fields[i][2:]
            if not tdata_size:
                tdata_size = "10b"
            break
    else:
        raise ValueError(f"data source and size not found for {model_path}")
    return training, architecture, tdata, tdata_source, tdata_size


def _model_idx(model_name):
    pref = model_name[:3]  # use first three letters are prefix for the index
    if pref not in _MODELS:
        _MODELS[pref] = {}
    _models = _MODELS[pref]
    if model_name not in _models:
        _models[model_name] = f"{pref}{len(_models)}"
    return _models[model_name]


def _is_hit_naive(confidence_scores, thresholds, ground_truth, label_sets):
    _is_cover = []
    for t in thresholds:
        for c in confidence_scores:
            if float(c) >= t:
                _is_cover.append(1)
                break
        else:
            _is_cover.append(0)
    assert len(_is_cover) == len(thresholds)
    for i, ls in enumerate(label_sets):
        if ground_truth in ls:
            return [int(float(confidence_scores[i]) > t) for t in thresholds], _is_cover
    return [0] * len(thresholds), _is_cover


def _load_json(model, data, thresholds):
    training, architecture, tdata, tdata_source, tdata_size = parse_file_name(
        Path(model).name
    )

    records = []
    for image_id, image_data in data.items():
        label = image_data["concept"]
        hits, covers = _is_hit_naive(
            image_data["confidence_scores"],
            thresholds,
            label,
            image_data["raw_prediction"],
        )
        houshold = image_id.split("/")[1]
        image_checksum = image_data["url"].split("/")[-1].split(".")[0].split("-")[-1]
        for t, hit, cover in zip(thresholds, hits, covers):
            model_name = f"{training} {architecture} {tdata} t={t}"
            records.append(
                (
                    training,
                    architecture,
                    tdata,
                    tdata_source,
                    tdata_size,
                    t,
                    _model_idx(model_name),
                    model_name,
                    image_data["region"],
                    houshold,
                    image_data["country"],
                    float(image_data["income"]),
                    image_checksum,
                    label,
                    hit,
                    cover,
                )
            )
    return pd.DataFrame.from_records(records, columns=COLUMNS)


def parse_all(datapath):
    _alldata = []
    for model in Path(datapath).glob("*.json"):
        with open(Path(datapath, model)) as f:
            print(model)
            this_data = _load_json(model, json.load(f), thresholds=THRESHOLDS)
            if this_data is not None:
                _alldata.append(this_data)
    return pd.concat(_alldata, ignore_index=True)


def add_log_income(results):
    """log income, bucket log income, income rank by localization, income rank by region by label"""
    new_results = results.copy()
    new_results["log_income"] = new_results["income"].apply(np.log)
    new_results["log_income_bckt"] = new_results["log_income"].apply(np.round)
    new_results["income_bucket"] = new_results["log_income"].apply(
        lambda x: INCOME_BUCKETS[np.round(x / 3)]
    )
    return new_results


def _plot_ir(data, *, model_name_string, figname):
    # plot for income and region
    _data = data.copy().sort_values(
        by=["trname", "archname", "sz", "income bucket", "region"], ascending=False
    )
    _data["new_model_name"] = _data.apply(
        lambda x: model_name_string.format(**x.to_dict()), axis=1
    )

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    seaborn.lineplot(
        data=_data,
        x="income bucket",
        y="hit",
        hue="new_model_name",
        style="new_model_name",
        ax=ax[0],
    )

    seaborn.lineplot(
        data=_data,
        x="region",
        y="hit",
        hue="new_model_name",
        style="new_model_name",
        ax=ax[1],
    )

    for i in range(2):
        ax[i].legend_.set_title(None)
        # set axis limits
        mini, maxi = ax[i].get_ylim()
        print(mini, maxi)
        if maxi - floor(10 * maxi) / 10 >= 0.05:
            maxi = ceil(maxi * 10) / 10
            ax[i].set_ylim(mini, maxi)
        ax[i].yaxis.set_major_locator(FixedLocator([x / 10 for x in range(11)]))
        ax[i].legend(loc="lower right")
        ax[i].grid(which="major", color="#d3d3d3", linestyle="dotted", linewidth=1)

    fig.tight_layout()
    fig.savefig(figname)

    # display dataframes side by side
    ldf = [
        _data.groupby(["income bucket", "new_model_name"])["hit"].mean().reset_index(),
        _data.groupby(["region", "new_model_name"])["hit"].mean().reset_index(),
    ]
    display_html(
        ("\xa0" * 10).join(
            [
                df.style.set_table_attributes("style='display:inline'")._repr_html_()
                for df in ldf
            ]
        ),
        raw=True,
    )


def generate_output_json_data(
    model_name, predictions_file, pred_confidence_scores_file, pred_img_indices_file
):
    my_model_in22k_subset_dollar_street_full_finetuned = {
        "model_name": model_name,
        "image_paths": "https://dl.fbaipublicfiles.com/vissl/fairness/dollarstreet_in22k_cls_overlapped_images.npy",
        "targets": "https://dl.fbaipublicfiles.com/vissl/fairness//subset_dollarstreet_in22k_cls_overlapped_labels.npy",
        "metadata": "https://dl.fbaipublicfiles.com/vissl/fairness/metadata_full_dollar_street.json",
        "id_to_label_map": "https://dl.fbaipublicfiles.com/vissl/fairness/in22k_cls_idx_to_dollar_street_labels_map.json",
        "predictions": predictions_file,
        "pred_img_indices": pred_img_indices_file,
        "pred_confidence_scores": pred_confidence_scores_file,
    }

    (
        my_model_output_attributes_acc_map,
        my_model_output_metadata_map,
    ) = generate_dollar_street_analysis(
        my_model_in22k_subset_dollar_street_full_finetuned,
        topk=PRED_TOPK,
        confidence_threshold=PRED_CONFIDENCE_THRESHOLD,
    )
    print(list(my_model_output_metadata_map.values())[:10])
    output_dir = "/tmp/dollar_street_models"
    output_file = f"{output_dir}/my_model_output_metadata_map.json"
    save_file(my_model_output_metadata_map, output_file)
    test_data_save = load_file(output_file)
    print(len(test_data_save))
    return output_dir


def calculate_metrics(output_dir):
    # reorder regions
    raw_results = parse_all(datapath=output_dir)

    # compressing results:
    # - removing label info
    # - for each image, counting hit and cover as max over possible labels
    # - average hits and covers over households.
    # image
    # new_results = results.groupby(by=COLUMNS_COMPRESSED + ["image"])[["hit", "cover"]].max().reset_index()
    # households
    raw_results_hh = (
        raw_results.groupby(by=COLUMNS_COMPRESSED)[["hit", "cover"]]
        .mean()
        .reset_index()
    )

    # income buckets
    raw_results = add_log_income(raw_results)
    raw_results_hh = add_log_income(raw_results_hh)

    raw_results["region"] = raw_results["region"].astype(_region_type)
    raw_results["archname"] = (
        raw_results["architecture"].apply(lambda x: ARCH_NAMES[x]).astype(_model_type)
    )
    raw_results["trname"] = (
        raw_results["training"]
        .apply(lambda x: TRAINING_NAMES[x])
        .astype(_training_type)
    )
    raw_results["sz"] = (
        raw_results["data_size"].apply(lambda x: SIZE_NAMES[x]).astype(_size_type)
    )
    raw_results["income bucket"] = raw_results["income_bucket"].astype(_income_type)

    print(raw_results_hh["architecture"])
    raw_results_hh["region"] = raw_results_hh["region"].astype(_region_type)
    raw_results_hh["archname"] = (
        raw_results_hh["architecture"]
        .apply(lambda x: ARCH_NAMES[x])
        .astype(_model_type)
    )
    raw_results_hh["trname"] = (
        raw_results_hh["training"]
        .apply(lambda x: TRAINING_NAMES[x])
        .astype(_training_type)
    )
    raw_results_hh["sz"] = (
        raw_results_hh["data_size"].apply(lambda x: SIZE_NAMES[x]).astype(_size_type)
    )
    raw_results_hh["income bucket"] = raw_results_hh["income_bucket"].astype(
        _income_type
    )

    # model size ig1b
    _plot_ir(
        raw_results_hh.query('training=="seer" & data_size=="1b"').sort_values(
            by="archname"
        ),
        model_name_string="{trname} - {archname}",
        figname="/tmp/dollar_street_models/my-model.png",
    )


if __name__ == "__main__":
    model_name = "Sup RN-50 (torchvision) IN1K"
    predictions_file = "/path/to/rank0_test_heads_predictions.npy"
    pred_confidence_scores_file = "/path/to/rank0_test_heads_conf_scores.npy"
    pred_img_indices_file = "/path/to/rank0_test_heads_inds.npy"

    output_dir = generate_output_json_data(
        model_name, predictions_file, pred_confidence_scores_file, pred_img_indices_file
    )
    calculate_metrics(output_dir)
