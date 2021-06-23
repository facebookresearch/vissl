#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This benchmark suite script launches a benchmark suite scheduler slurm job.
# The job takes an absolute json config path (see benchmark_suite_scheduler_template.json for info)
# The job continuously monitors training benchmarks, and dynamically launches evaluation jobs
# and amalgamates the results.

######################### EXAMPLE USAGE #################################

# cd into vissl root directory.
#
# bash ./dev/launch_benchmark_suite_scheduler_slurm.sh /path/to/benchmark_suite_scheduler.json

# See benchmark_suite_scheduler_template.json or for config information or slurm_evaluator.py for class structure.
######################### INPUT PARAMS ##################################

FILE=( "$@" )

####################### setup experiment dir ###################################

# create a temporary experiment folder to run the SLURM job in isolation
RUN_ID=$(date +'%Y-%m-%d-%H-%M-%S')
EXP_ROOT_DIR="/checkpoint/$USER/vissl/$RUN_ID"

echo "EXP_ROOT_DIR: $EXP_ROOT_DIR"
echo "CONFIG_FILE: ${FILE[0]}"

rm -rf "$EXP_ROOT_DIR"
mkdir -p "$EXP_ROOT_DIR"
cp -r . "$EXP_ROOT_DIR"

####################### setup experiment dir ###################################
export PYTHONPATH="$EXP_ROOT_DIR/:$PYTHONPATH"
python -u "$EXP_ROOT_DIR/tools/launch_benchmark_suite_scheduler_slurm.py" \
    "${FILE[@]}"
