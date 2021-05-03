# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from vissl.data.datasets.coco import get_coco_imgs_labels_info
from vissl.data.datasets.pascal_voc import get_voc_images_labels_info


__all__ = ["get_coco_imgs_labels_info", "get_voc_images_labels_info"]
