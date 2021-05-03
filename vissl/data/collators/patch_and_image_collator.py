# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from vissl.data.collators import register_collator


@register_collator("patch_and_image_collator")
def patch_and_image_collator(batch):
    """
    This collator is used in PIRL approach.

    batch contains two keys "data" and "label".
        - data is a list of N+1 elements.
          1st element is the "image" and remainder N
          are patches.
        - label is an integer (image index in the dataset)

    We collate this to
        image: batch_size tensor containing images
        patches: N * batch_size tensor containing patches
    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"
    batch_size = len(batch)
    data = [x["data"] for x in batch]
    # labels are repeated N+1 times but they are the same
    labels = [x["label"][0] for x in batch]
    labels = torch.LongTensor(labels).squeeze()

    # data valid is repeated N+1 times but they are the same
    data_valid = torch.BoolTensor([x["data_valid"][0] for x in batch])
    images = torch.stack([data[i][0] for i in range(batch_size)])
    patch_list = []
    for idx in range(batch_size):
        patch_list.extend(data[idx][1:])
    patches = torch.stack(patch_list)

    output_batch = {
        "images": [images],
        "patches": [patches],
        "label": [labels],
        "data_valid": [data_valid],
    }

    return output_batch
