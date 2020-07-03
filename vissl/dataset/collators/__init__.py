# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torch.utils.data.dataloader import default_collate
from vissl.dataset.collators.flatten_collator import flatten_collator
from vissl.dataset.collators.multires_collator import multires_collator
from vissl.dataset.collators.patch_and_image_collator import patch_and_image_collator
from vissl.dataset.collators.siamese_collator import siamese_collator


COLLATORS_MAP = {
    "default": default_collate,
    "flatten_collator": flatten_collator,
    "siamese_collator": siamese_collator,
    "patch_and_image_collator": patch_and_image_collator,
    "multires_collator": multires_collator,
}


def get_collator(name):
    assert name in COLLATORS_MAP, "Unknown collator"
    return COLLATORS_MAP[name]


__all__ = ["get_collator"]
