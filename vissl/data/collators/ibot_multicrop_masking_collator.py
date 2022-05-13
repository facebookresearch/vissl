# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from vissl.data.collators import register_collator


@register_collator("ibot_multicrop_masking_collator")
def ibot_multicrop_masking_collator(batch: dict):
    """
    This collator is used in iBOT approach (https://arxiv.org/pdf/2111.07832.pdf).

    The multiple views will be assembled like so:
    {
        "global_views": [global crops 1 (batch_size)] + [global crops 2 (batch_size)]
        "local_views":  [local crops 1 (batch_size)] + [local crops 2 (batch_size)]
    }
    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"
    assert "mask" in batch[0], "mask not found in sample"

    data = [x["data"] for x in batch]
    masks = [x["mask"] for x in batch]
    labels = [torch.tensor(x["label"]) for x in batch]
    data_valid = [torch.tensor(x["data_valid"]) for x in batch]
    data_idx = [torch.tensor(x["data_idx"]) for x in batch]
    num_duplicates, num_images = len(data[0]), len(data)

    # The data will be arranged such that:
    # - the first crops will be together, then the second crops, etc
    # - global views and local views will be separated in two groups
    big_crop_size = max(
        max(data[idx][pos].shape[1:])
        for pos in range(num_duplicates)
        for idx in range(num_images)
    )
    global_views = []
    local_views = []
    global_masks = []
    output_label = []
    output_data_valid = []
    output_data_idx = []
    for pos in range(num_duplicates):
        for idx in range(num_images):
            image_shape = data[idx][pos].shape[1:]
            if max(image_shape) == big_crop_size:
                global_views.append(data[idx][pos])
                global_masks.append(masks[idx][pos])
            else:
                local_views.append(data[idx][pos])
            output_label.append(labels[idx][pos])
            output_data_valid.append(data_valid[idx][pos])
            output_data_idx.append(data_idx[idx][pos])

    # Return the output batch, enriched with the mask information
    output_batch = {
        "global_views": [torch.stack(global_views)],
        "local_views": [torch.stack(local_views)],
        "mask": [torch.stack(global_masks)],
        "label": [torch.stack(output_label)],
        "data_valid": [torch.stack(output_data_valid)],
        "data_idx": [torch.stack(output_data_idx)],
    }
    return output_batch
