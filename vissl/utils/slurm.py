#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os


def get_node_id(node_id: int):
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is not None:
        node_name = str(os.environ["SLURMD_NODENAME"])
        node_id = int(os.environ.get("SLURM_NODEID"))
        logging.info(f"SLURM job: node_name: {node_name}, node_id: {node_id}")
    return node_id


def get_slurm_dir(input_dir: str):
    output_dir = input_dir
    if "SLURM_JOBID" in os.environ:
        output_dir = f"{input_dir}/{os.environ['SLURM_JOBID']}"
    return output_dir
