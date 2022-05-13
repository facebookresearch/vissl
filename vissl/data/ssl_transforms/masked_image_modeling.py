# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("MaskedImageModeling")
class MaskedImageModeling(ClassyTransform):
    """Augmentation used to generate a mask for Vision Transformers
    Typically used in approaches such as iBOT (https://arxiv.org/pdf/2111.07832.pdf)
    """

    def __init__(
        self,
        pred_ratio_mean: List[float],
        pred_ratio_var: List[float],
        patch_size: int,
        log_aspect_ratio: Tuple[float, float],
    ):
        self.pred_ratio_mean = pred_ratio_mean
        self.pred_ratio_var = pred_ratio_var
        self.patch_size = patch_size
        self.log_aspect_ratio = log_aspect_ratio

    def __call__(self, x: torch.Tensor):
        image_shape = x.shape[1:]
        pred_ratio = get_pred_ratio(self.pred_ratio_mean, self.pred_ratio_var)
        mask_shape = (
            image_shape[0] // self.patch_size,
            image_shape[1] // self.patch_size,
        )
        mask = create_mask(
            high=int(mask_shape[0] * mask_shape[1] * pred_ratio),
            mask_shape=mask_shape,
            log_aspect_ratio=self.log_aspect_ratio,
        )
        return {"data": x, "mask": mask}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskedImageModeling":
        return cls(
            pred_ratio_mean=config.get("pred_ratio_mean", [0.3]),
            pred_ratio_var=config.get("pred_ratio_var", [0.0]),
            patch_size=config.get("patch_size", 16),
            log_aspect_ratio=config.get("log_aspect_ratio", (0.3, 1 / 0.3)),
        )


def get_pred_ratio(pred_ratio_mean: List[float], pred_ratio_var: List[float]):
    idx = random.randint(0, len(pred_ratio_mean) - 1)
    prm = pred_ratio_mean[idx]
    prv = pred_ratio_var[idx]
    return random.uniform(prm - prv, prm + prv) if prv > 0 else prm


def create_mask(
    high: int, mask_shape: Tuple[int, int], log_aspect_ratio: Tuple[float, float]
) -> torch.Tensor:
    """Create a mask for the global view
    Copy-pasted and adapted from iBOT (https://arxiv.org/pdf/2111.07832.pdf) official code:
    https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/loader.py#L67
    """
    H, W = mask_shape
    mask = np.zeros(mask_shape, dtype=bool)
    mask_count = 0
    while mask_count < high:
        max_mask_patches = high - mask_count

        delta = 0
        for _attempt in range(10):
            low = (min(mask_shape) // 3) ** 2
            target_area = random.uniform(low, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

            if delta > 0:
                break

        if delta == 0:
            break
        else:
            mask_count += delta

    return torch.from_numpy(mask)
