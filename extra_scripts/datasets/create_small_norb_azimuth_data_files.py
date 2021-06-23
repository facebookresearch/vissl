# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os


def import_norb_elevation_module():
    """
    A convoluted way to import 'create_small_norb_elevation_data_files.py' and reuse the code it contains
    """
    script_path = os.path.split(__file__)[0]
    norb_elevation_path = os.path.join(
        script_path, "create_small_norb_elevation_data_files.py"
    )
    spec = importlib.util.spec_from_file_location(
        "create_small_norb_elevation_data_files", norb_elevation_path
    )
    norb_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norb_module)
    return norb_module


if __name__ == "__main__":
    """
    Example usage:

    ```
    python extra_scripts/datasets/create_small_norb_azimuth_data_files.py -i /path/to/small_norb/ -o /output_path/to/small_norb
    ```
    """
    norb_module = import_norb_elevation_module()
    args = norb_module.get_argument_parser().parse_args()
    if args.download:
        norb_module.download_dataset(args.input)
    norb_module.create_norm_elevation_dataset(
        input_path=args.input,
        output_path=args.output,
        target_transform=norb_module.parse_azimuth,
    )
