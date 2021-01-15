# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from vissl.data.collators import register_collator


@register_collator("siamese_collator")
def siamese_collator(batch):
    """
    This collator is used in Jigsaw approach.

    Input:
        batch: Example
                batch = [
                    {"data": [img1,], "label": [lbl1, ]},        #img1
                    {"data": [img2,], "label": [lbl2, ]},        #img2
                    .
                    .
                    {"data": [imgN,], "label": [lblN, ]},        #imgN
                ]

                where:
                    img{x} is a tensor of size: num_towers x C x H x W
                    lbl{x} is an integer

    Returns: Example output:
                output = [
                    {
                        "data": torch.tensor([img1_0, ..., imgN_0]) ..
                    },
                ]
                where the output is of dimension: (N * num_towers) x C x H x W
    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"
    num_data_sources = len(batch[0]["data"])
    batch_size = len(batch)
    data = [x["data"] for x in batch]
    labels = [x["label"] for x in batch]
    data_valid = torch.BoolTensor([x["data_valid"][0] for x in batch])

    output_data, output_label = [], []
    for idx in range(num_data_sources):
        # each image is of shape: num_towers x C x H x W
        # num_towers x C x H x W -> N x num_towers x C x H x W
        idx_data = torch.stack([data[i][idx] for i in range(batch_size)])
        idx_labels = [labels[i][idx] for i in range(batch_size)]
        batch_size, num_siamese_towers, channels, height, width = idx_data.size()
        # N x num_towers x C x H x W -> (N * num_towers) x C x H x W
        idx_data = idx_data.view(
            batch_size * num_siamese_towers, channels, height, width
        )
        output_data.append(idx_data)
        should_flatten = False
        if idx_labels[0].ndim == 1:
            should_flatten = True
        idx_labels = torch.stack(idx_labels).squeeze()
        if should_flatten:
            idx_labels = idx_labels.flatten()
        output_label.append(idx_labels)

    output_batch = {
        "data": output_data,
        "label": output_label,
        "data_valid": [data_valid],
    }
    return output_batch
