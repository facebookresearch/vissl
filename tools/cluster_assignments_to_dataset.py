# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert a 'cluster_assignment.torch' file (obtained for example
by extracting the SwAV cluster assignment on a dataset) to a
disk_filelist dataset.

The resulting disk_filelist can be used to trained another model

Example usage:

    ```
    python extra_scripts/cluster_assignments_to_dataset.py
        -i /path/to/cluster_assignments.torch
        -o /path/to/dataset/folder
    ```
"""

import argparse

from vissl.utils.cluster_utils import ClusterAssignmentLoader


def get_argument_parser():
    """
    List of arguments supported by the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="Path to 'cluster_assignment.torch'"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Where to write the disk_filelist dataset"
    )
    return parser


def main():
    args = get_argument_parser().parse_args()
    loaded_assignments = ClusterAssignmentLoader.load_cluster_assigment(args.input)
    ClusterAssignmentLoader.save_cluster_assignment_as_dataset(
        output_dir=args.output, assignments=loaded_assignments
    )


if __name__ == "__main__":
    main()
