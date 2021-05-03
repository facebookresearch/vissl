# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from vissl.data.collators import register_collator


@register_collator("simclr_collator")
def simclr_collator(batch):
    """
    This collator is used in SimCLR approach.

    The collators collates the batch for the following input (each image has k-copies):
        input: [[img1_0, ..., img1_k], [img2_0, ..., img2_k], ..., [imgN_0, ..., imgN_k]]
        output: [img1_0, img2_0, ....., img1_1, img2_1,...]


    Input:
        batch: Example
                batch = [
                    {"data": [img1_0, ..., img1_k], "label": [lbl1, ]},        #img1
                    {"data": [img2_0, ..., img2_k], "label": [lbl2, ]},        #img2
                    .
                    .
                    {"data": [imgN_0, ..., imgN_k], "label": [lblN, ]},        #imgN
                ]

                where:
                    img{x} is a tensor of size: C x H x W
                    lbl{x} is an integer

    Returns: Example output:
                output = [
                    {
                        "data": torch.tensor([img1_0, img2_0, ....., img1_1, img2_1,...]) ..
                    },
                ]
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
