# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

import torch
from vissl.data.collators import register_collator


@register_collator("moco_collator")
def moco_collator(batch: List[Dict[str, Any]]) -> Dict[str, List[torch.Tensor]]:
    """
    This collator is specific to MoCo approach http://arxiv.org/abs/1911.05722

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
                        "data": torch.tensor([img1_0, ..., img1_k], [img2_0, ..., img2_k]) ..
                    },
                ]

             Dimensions become [num_positives x Batch x C x H x W]
    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"

    data = [torch.stack(x["data"]) for x in batch]
    labels = [torch.tensor(x["label"]) for x in batch]
    data_valid = [torch.tensor(x["data_valid"]) for x in batch]
    data_idx = [torch.tensor(x["data_idx"]) for x in batch]

    output_batch = {
        "data": [torch.stack(data).squeeze()[:, 0, :, :, :].squeeze()],  # encoder
        "data_momentum": [
            torch.stack(data).squeeze()[:, 1, :, :, :].squeeze()
        ],  # momentum encoder
        "label": [torch.stack(labels).squeeze()],
        "data_valid": [torch.stack(data_valid).squeeze()],
        "data_idx": [torch.stack(data_idx).squeeze()],
    }

    return output_batch
