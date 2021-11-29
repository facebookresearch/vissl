# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from vissl.data.collators import register_collator


@register_collator("identity_collator")
def identity_collator(batch):
    """
    This collator simply returns the input batch.
    """
    return batch
