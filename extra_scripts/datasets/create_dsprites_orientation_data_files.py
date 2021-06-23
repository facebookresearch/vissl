# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import math
import os

import numpy as np


def import_dsprite_location_module():
    """
    Import the 'create_dsprites_location_data_files.py' script to reuse the code it contains
    """
    script_path = os.path.split(__file__)[0]
    module_name = "create_dsprites_location_data_files"
    module_path = os.path.join(script_path, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_binned_orientation(latents: np.ndarray) -> int:
    """
    Return orientation of the sprite, binned in 16 different classes.

    The original x position is given as a float between 0.0 and 2*pi.
    We transform it so that each bucket is as evenly filled as possible.
    """
    max_angle = 2 * math.pi
    nb_buckets = 16
    orientation = latents[3]
    binned_orientation = np.clip(
        np.floor(orientation / max_angle * nb_buckets), 0, nb_buckets - 1
    )
    return int(binned_orientation)


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_dsprites_orientation_data_files.py -i /path/to/dsprites/ -o /output_path/to/dsprites_or
    ```
    """
    dsprite_module = import_dsprite_location_module()
    args = dsprite_module.get_argument_parser().parse_args()
    if args.download:
        dsprite_module.download_dataset(args.input)
    dsprite_module.create_dataset(
        input_folder=args.input,
        output_folder=args.output,
        target_transform=get_binned_orientation,
    )
