# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from vissl.data.collators import register_collator


def _blend_images(images, mixing_factor):
    assert len(images) == 2, "mixup is only supported for 2 images at a time currently"
    # multiply the tensors with the respective mixing value
    images[0].mul_(mixing_factor)
    images[1].mul_(1 - mixing_factor)
    images[0].add_(images[1])
    return images[0]


@register_collator("multicrop_mixup_collator")
def multicrop_mixup_collator(batch):
    """
    This collator is used to mix-up 2 images at a time. 2*N input images becomes N images
    This collator can handle multi-crop input. For each crop, it mixes-up the corresponding
    crop of the next image.

    Input:
        batch: Example
                batch = [
                    {"data" : [img1_0, ..., img1_k], ..},
                    {"data" : [img2_0, ..., img2_k], ...},
                    ...
                    {"data" : [img2N_0, ..., img2N_k], ...},
                ]

    Returns: Example output:
                output = [
                    {
                        "data": [
                            torch.tensor([img1_2_0, ..., img1_2_k]),
                            torch.tensor([img3_4_0, ..., img3_4_k])
                            ...
                        ]
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

    # we apply the mixup now: (2 * N) images input -> N images
    beta = 0.2
    mixing_factor = np.random.beta(beta, beta)
    output_data, output_label, output_data_valid, output_data_idx = [], [], [], []
    for pos in range(num_duplicates):
        _output_data = []
        for idx in range(0, num_images, 2):
            _output_data.append(
                _blend_images(
                    images=[data[idx][pos], data[idx + 1][pos]],
                    mixing_factor=mixing_factor,
                )
            )
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
