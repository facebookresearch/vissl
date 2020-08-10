# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from vissl.data.collators import register_collator


@register_collator("simclr_collator")
def simclr_collator(batch):
    """
    The collators collates the batch for the following input:
    input: [[img1_0, ..., img1_k], [img2_0, ..., img2_k], ..., [imgN_0, ..., imgN_k]]
    output: [img1_0, img2_0, ....., img1_1, img2_1,...]
    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"

    data = [x["data"] for x in batch]
    labels = [torch.tensor(x["label"]) for x in batch]
    data_valid = [torch.tensor(x["data_valid"]) for x in batch]
    data_idx = [torch.tensor(x["data_idx"]) for x in batch]
    num_positives, num_images = len(data[0]), len(data)

    output_data, output_label, output_data_valid, output_data_idx = [], [], [], []
    for pos in range(num_positives):
        for idx in range(num_images):
            output_data.append(data[idx][pos])
            output_label.append(labels[idx][pos])
            output_data_valid.append(data_valid[idx][pos])
            output_data_idx.append(data_idx[idx][pos])

    output_batch = {
        "data": [torch.stack(output_data)],
        "label": [torch.stack(output_label)],
        "data_valid": [torch.stack(output_data_valid)],
        "data_idx": [torch.stack(output_data_idx)],
    }
    return output_batch
