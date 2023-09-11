import numpy as np
import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from typing import Tuple, List, Dict, Any
from random import randint

@register_transform("RandomCompMasking")
class RandomCompMasking(ClassyTransform):
    """
        Create masks for input sample.
    """

    def __init__(self,
                masks_shapes:List[List[Tuple[int, int]]]=[[(6,6)], [(6,6), (12,3), (3,12)]],
                padding:Tuple[int, int]=(0, 0),
                out_shape:Tuple[int, int]=(14, 14),
                masking_mode="primary_secondary"):
        """
        Args:
            mask_areas: list of fractions of image areas each mash covers
            padding: padding for mask sampling
            patch_size: patch_size
            masking_mode: mode describing positional relations between masks
            ...
        """
        assert all([shape[0]*shape[1] == mask_shapes[0][0]*mask_shapes[0][1] for mask_shapes in masks_shapes for shape in mask_shapes]), "All shapes given masks can be sampled have to cover the same number of patches."
        self.masks_shapes = masks_shapes
        self.padding = padding
        self.out_shape = out_shape
        self.masking_mode = masking_mode

    def primary_secondary(self):
        mask0_shapes, mask1_shapes = self.masks_shapes[:2]
        y_padding, x_padding = self.padding
        H, W = self.out_shape


        fitting_masks = [(m0, m1) for m0 in mask0_shapes for m1 in mask1_shapes
                    if m0[0]+m1[0]+y_padding<=H and max(m0[1],m1[1]) <= W-x_padding
                    or m0[1]+m1[1]+x_padding<=W and max(m0[0],m1[0]) <= H-y_padding]
        (mask0_y, mask0_x), (mask1_y, mask1_x) = fitting_masks[randint(0, len(fitting_masks)-1)]
    
        if mask1_x + x_padding + mask0_x <= W and mask1_y + y_padding + mask0_y <= H:
            dice = randint(0,3)
        elif mask1_x + x_padding + mask0_x <= W:
            dice = randint(0,1)
        elif mask1_y + y_padding + mask0_y <= H:
            dice = randint(2,3)
        else:
            assert False

        left_padding = max(x_padding, (dice == 0) * mask1_x)
        right_padding = max(x_padding, (dice == 1) * mask1_x)
        up_padding = max(y_padding, (dice == 2) * mask1_y)
        down_padding = max(y_padding, (dice == 3) * mask1_y)

        mask0 = np.zeros((H, W), dtype=bool)
        
        mask0_y_corner = randint(down_padding, H-up_padding-mask0_y) if down_padding <= H-up_padding-mask0_y else randint(0, H-mask0_y)
        mask0_x_corner = randint(left_padding, W-right_padding-mask0_x) if left_padding <= W-right_padding-mask0_x else randint(0, W-mask0_x)
            
        for i in range(mask0_y):
            for j in range(mask0_x):
                mask0[mask0_y_corner + i, mask0_x_corner + j] = 1

        space_left = (H, mask0_x_corner)
        space_right = (H, W-mask0_x_corner-mask0_x)
        space_up = (mask0_y_corner, W)
        space_down = (H-mask0_y_corner-mask0_y, W)
        space = [space_left, space_right, space_up, space_down]

        fits = [i for i, (y, x) in enumerate(space) if mask1_y <= y and mask1_x <= x]
        dice = randint(0, len(fits)-1)
        fit_case = fits[dice]

        if fit_case == 0:
            sampling_range_x = (x_padding, mask0_x_corner-mask1_x) if x_padding <= mask0_x_corner-mask1_x else (0, mask0_x_corner-mask1_x)
            sampling_range_y = (y_padding, H-y_padding-mask1_y) if y_padding <= H-y_padding-mask1_y else (0, H-mask1_y)
        elif fit_case == 1:
            sampling_range_x = (mask0_x_corner+mask0_x, W-x_padding-mask1_x) if mask0_x_corner+mask0_x <= W-x_padding-mask1_x else (mask0_x_corner+mask0_x, W-mask1_x)
            sampling_range_y = (y_padding, H-y_padding-mask1_y) if y_padding <= H-y_padding-mask1_y else (0, H-mask1_y)
        elif fit_case == 2:
            sampling_range_x = (x_padding, W-x_padding-mask1_x) if x_padding <= W-x_padding-mask1_x else (0, W-mask1_x)
            sampling_range_y = (y_padding, mask0_y_corner-mask1_y) if y_padding <= mask0_y_corner-mask1_y else (0, mask0_y_corner-mask1_y)
        elif fit_case == 3:
            sampling_range_x = (x_padding, W-x_padding-mask1_x) if x_padding <= W-x_padding-mask1_x else (0, W-mask1_x)
            sampling_range_y = (mask0_y_corner+mask0_y, H-y_padding-mask1_y) if mask0_y_corner+mask0_y <= H-y_padding-mask1_y else (mask0_y_corner+mask0_y, H-mask1_y)
        else:
            assert False, fit_case

        mask1_x_corner = randint(*sampling_range_x)
        mask1_y_corner = randint(*sampling_range_y)

        mask1 = np.zeros((H, W), dtype=bool)
        for i in range(mask1_y):
            for j in range(mask1_x):
                mask1[mask1_y_corner + i, mask1_x_corner + j] = 1
        return mask0, mask1

    def __call__(self, image: torch.Tensor):
        return image, {"masks": getattr(self, self.masking_mode)()}
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RandomCompMasking":
        """
        Instantiates CompositionalMasks transform from configuration.

        Args:
            config (Dict): arguments for the transform

        Returns:
            CompositionalMasks transform instance.
        """
        padding = config.get("padding", (0, 0))
        out_shape = config.get("patch_size", (14, 14))
        masking_mode = config['masking_mode']
        masks_shapes = config['masks_shapes']
        assert masking_mode == "primary_secondary"
        return cls(masks_shapes, padding, out_shape, masking_mode)