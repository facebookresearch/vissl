#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

######################### EXAMPLE USAGE #################################
#
# ./dev/launch_slurm.sh
#    config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear
#    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/checkpoint/user/checkpoint.torch
#
# Configuration for SLURM can be provided as additional hydra overrides:
#
# ./dev/launch_slurm.sh
#    config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear
#    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/checkpoint/user/checkpoint.torch
#    config.SLURM.NAME=linear_evaluation
#    config.SLURM.COMMENT=linear_evaluation_on_simclr
#    config.SLURM.PARTITION=learnfair

######################### INPUT PARAMS ##################################

CFG=( "$@" )

####################### setup experiment dir ###################################

# create a temporary experiment folder to run the SLURM job in isolation
RUN_ID=$(date +'%Y-%m-%d-%H-%M-%S')
EXP_ROOT_DIR="/checkpoint/$USER/vissl/$RUN_ID"
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$EXP_ROOT_DIR/checkpoints/"}

echo "EXP_ROOT_DIR: $EXP_ROOT_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"

rm -rf $EXP_ROOT_DIR
mkdir -p "$EXP_ROOT_DIR"
mkdir -p "$CHECKPOINT_DIR"
cp -r . $EXP_ROOT_DIR

####################### launch script #########################################

export PYTHONPATH="$EXP_ROOT_DIR/:$PYTHONPATH"
python -u "$EXP_ROOT_DIR/tools/run_distributed_engines.py" \
  "${CFG[@]}" \
  hydra.run.dir="$EXP_ROOT_DIR" \
  config.SLURM.USE_SLURM=true \
  config.SLURM.LOG_FOLDER="$EXP_ROOT_DIR" \
  config.CHECKPOINT.DIR="$CHECKPOINT_DIR"
