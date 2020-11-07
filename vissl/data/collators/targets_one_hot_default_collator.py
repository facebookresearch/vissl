# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# from typing import List

import torch
from vissl.data.collators import register_collator


def convert_to_one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.
    """
    assert (
        torch.max(targets).item() < num_classes
    ), "Class Index must be less than number of classes"
    one_hot_targets = torch.zeros(
        (1, num_classes), dtype=torch.long, device=targets.device
    )
    one_hot_targets.scatter_(1, targets.view(1, -1).long(), 1)
    return one_hot_targets.squeeze()


@register_collator("targets_one_hot_default_collator")
def targets_one_hot_default_collator(batch, num_classes: int):
    """
    The collators collates the batch for the following input:
    Input:
        input : [[img0, ..., imgk]]
        label: [[1, 3, 6], [1, 5], .....]
    Output:
        output: [img0, img0, .....,]
        label: [[0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0], ....]
    """
    assert num_classes > 0, "num_classes not specified for the collator"
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"
    assert len(batch[0]["data"]) == 1, (
        "This collator supports only 1 data source. "
        "Please extend it to support many data sources."
    )
    data = torch.stack([x["data"][0] for x in batch])
    data_valid = torch.stack([torch.tensor(x["data_valid"][0]) for x in batch])
    data_idx = torch.stack([torch.tensor(x["data_idx"][0]) for x in batch])
    labels = [torch.tensor(x["label"][0]) for x in batch]

    output_labels = []
    for idx in range(data.shape[0]):
        output_labels.append(convert_to_one_hot(labels[idx], num_classes))

    output_batch = {
        "data": [data],
        "label": [torch.stack(output_labels)],
        "data_valid": [data_valid],
        "data_idx": [data_idx],
    }
    return output_batch
