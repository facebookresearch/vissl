# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from vissl.data.datasets.coco import get_coco_imgs_labels_info
from vissl.data.datasets.pascal_voc import get_voc_images_labels_info


__all__ = ["get_coco_imgs_labels_info", "get_voc_images_labels_info"]
