# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from vissl.data.collators import register_collator


@register_collator("multicrop_collator")
def multicrop_collator(batch):
    """
    This collator is used in SwAV approach.

    The collators collates the batch for the following input (assuming k-copies of image):

    Input:
        batch: Example
                batch = [
                    {"data" : [img1_0, ..., img1_k], ..},
                    {"data" : [img2_0, ..., img2_k], ...},
                    ...
                ]

    Returns: Example output:
                output = [
                    {
                        "data": torch.tensor([img1_0, ..., imgN_0], [img1_k, ..., imgN_k]) ..
                    },
                ]
    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"

    data = [x["data"] for x in batch]
    labels = [torch.tensor(x["label"]) for x in batch]
    data_valid = [torch.tensor(x["data_valid"]) for x in batch]
    data_idx = [torch.tensor(x["data_idx"]) for x in batch]
    num_duplicates, num_images = len(data[0]), len(data)

    output_data, output_label, output_data_valid, output_data_idx = [], [], [], []
    for pos in range(num_duplicates):
        _output_data = []
        for idx in range(num_images):
            _output_data.append(data[idx][pos])
            output_label.append(labels[idx][pos])
            output_data_valid.append(data_valid[idx][pos])
            output_data_idx.append(data_idx[idx][pos])
        output_data.append(torch.stack(_output_data))

    output_batch = {
        "data": [output_data],
        "label": [torch.stack(output_label)],
        "data_valid": [torch.stack(output_data_valid)],
        "data_idx": [torch.stack(output_data_idx)],
    }

    return output_batch
