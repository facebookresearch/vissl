# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torchvision.transforms as pth_transforms
from classy_vision.generic.registry_utils import import_all_modules
from vissl.data.ssl_transforms.ssl_transforms_wrapper import (
    DEFAULT_TRANSFORM_TYPES,
    SSLTransformsWrapper,
)


def get_transform(input_transforms_list):
    """
    Given the list of user specified transforms, return the
    torchvision.transforms.Compose() version of the transforms. Each transform
    in the composition is SSLTransformsWrapper which wraps the original
    transforms to handle multi-modal nature of input.
    """
    output_transforms = []
    for transform_config in input_transforms_list:
        transform = SSLTransformsWrapper.from_config(
            transform_config, transform_types=DEFAULT_TRANSFORM_TYPES
        )
        output_transforms.append(transform)
    return pth_transforms.Compose(output_transforms)


FILE_ROOT = Path(__file__).parent

import_all_modules(FILE_ROOT, "vissl.data.ssl_transforms")

__all__ = ["get_transform"]
