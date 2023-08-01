import numpy as np
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from typing import Tuple, List, Dict, Any
from math import sqrt
from collections import namedtuple

InputKwargs = namedtuple("InputKwargs", ["input", "kwargs"])

@register_transform("CompositionalMasks")
class CompositionalMasks(ClassyTransform):
    """
        Create masks for input sample.
    """

    def __init__(self, mask_areas:List[float]=[0.2, 0.2], padding:Tuple[int, int]=(0.2, 0.2), patch_size:Tuple[int, int]=(16, 16), masking_mode="glued_sides"):
        """
        Args:
            mask_areas: list of fractions of image areas each mash covers
            padding: padding for mask sampling
            patch_size: patch_size
            masking_mode: mode describing positional relations between masks
            ...
        """
        self.mask_areas = mask_areas
        self.padding = padding
        self.patch_size = patch_size
        self.masking_mode = masking_mode

    def glued_sides(self, image_size):
            H, W = image_size[0] // self.patch_size[0], image_size[1] // self.patch_size[1]
            y_padding, x_padding = int(self.padding[0] * H), int(self.padding[1] * W) 
            mask0 = np.zeros((H, W), dtype=bool)
            mask0_sq_len = int(sqrt(self.mask_areas[0]*H*W))
            mask0_y_corner, mask0_x_corner = np.random.randint(y_padding, H-y_padding-mask0_sq_len, 1), np.random.randint(x_padding, W-x_padding-mask0_sq_len, 1)
            for i in range(mask0_sq_len):
                for j in range(mask0_sq_len):
                    mask0[mask0_y_corner + i, mask0_x_corner + j] = 1

            mask1 = np.zeros((H, W), dtype=bool)
            mask1_sq_len = int(sqrt(self.mask_areas[1]*H*W))
            mask1_corner_offset = np.random.randint(0, mask0_sq_len-mask1_sq_len, 1) if mask0_sq_len-mask1_sq_len > 0 else 0
            coin = np.random.randint(0,2,1) == 0
            if mask0_y_corner > 0.5 * (H-mask0_sq_len) and mask0_x_corner > 0.5 * (W-mask0_sq_len):
                if mask0_y_corner / H > mask0_x_corner / W or (mask0_y_corner / H == mask0_x_corner / W and coin):
                    mask1_y_corner = mask0_y_corner - mask1_sq_len
                    mask1_x_corner = mask0_x_corner + mask1_corner_offset
                else:
                    mask1_y_corner = mask0_y_corner + mask1_corner_offset
                    mask1_x_corner = mask0_x_corner - mask1_sq_len
            elif mask0_y_corner <= 0.5 * (H-mask0_sq_len) and mask0_x_corner <= 0.5 * (W-mask0_sq_len):
                if mask0_y_corner / H < mask0_x_corner / W or (mask0_y_corner / H == mask0_x_corner / W and coin):
                    mask1_y_corner = mask0_y_corner + mask0_sq_len
                    mask1_x_corner = mask0_x_corner + mask1_corner_offset
                else:
                    mask1_y_corner = mask0_y_corner + mask1_corner_offset
                    mask1_x_corner = mask0_x_corner + mask0_sq_len
            elif mask0_y_corner > 0.5 * (H-mask0_sq_len) and mask0_x_corner <= 0.5 * (W-mask0_sq_len):
                    mask1_y_corner = mask0_y_corner - mask1_sq_len
                    mask1_x_corner = mask0_x_corner + mask1_corner_offset
            elif mask0_y_corner <= 0.5 * (H-mask0_sq_len) and mask0_x_corner > 0.5 * (W-mask0_sq_len):
                    mask1_y_corner = mask0_y_corner + mask1_corner_offset
                    mask1_x_corner = mask0_x_corner - mask1_sq_len
            else:
                 assert False

            mask1_x_corner, mask1_y_corner = max(min(mask1_x_corner, 0), W), max(min(mask1_y_corner, 0), H)
            for i in range(mask1_sq_len):
                for j in range(mask1_sq_len):
                    try:
                        mask1[mask1_y_corner + i, mask1_x_corner + j] = 1
                    except:
                         continue

            return mask0, mask1

    def __call__(self, image: torch.Tensor):
        assert len(image.shape) == 3
        img_size = image.shape[1:]
        return InputKwargs(image, {"masks": self.__getattribute__(self.masking_mode)(img_size)})
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompositionalMasks":
        """
        Instantiates CompositionalMasks transform from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            CompositionalMasks transform instance.
        """
        mask_areas = config['mask_areas']
        padding = config.get("padding", (0.2, 0.2))
        patch_size = config.get("patch_size", (16, 16))
        masking_mode = config['masking_mode']
        return cls(mask_areas, padding, patch_size, masking_mode)