#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

######################### INPUT PARAMS ##################################

EXPT_NAME=${EXPT_NAME-'unnamed'}
PARTITION=${PARTITION-'learnfair'}
COMMENT=${COMMENT-''}
RUN_ID=$(date +'%Y-%m-%d-%H:%M:%S')
CFG=( "$@" )

echo "EXPT_NAME: $EXPT_NAME"
echo "COMMENT: $COMMENT"
echo "PARTITION: $PARTITION"

####################### setup experiment dir ###################################

# create the experiments folder
EXP_ROOT_DIR="/checkpoint/$USER/vissl/$RUN_ID/$EXPT_NAME"
RUN_SCRIPT="$EXP_ROOT_DIR/tools/run_distributed_on_slurm.py"
CHECKPOINT_DIR="$EXP_ROOT_DIR/checkpoints/"

echo "EXP_ROOT_DIR: $EXP_ROOT_DIR"
echo "CHECKPOINT_DIR: $CHECKPOINT_DIR"

# make the exp_dir and clone the current code inside it
rm -rf $EXP_ROOT_DIR
mkdir -p "$EXP_ROOT_DIR"
mkdir -p "$CHECKPOINT_DIR"
cp -r . $EXP_ROOT_DIR

####################### launch script #########################################

python -u "$RUN_SCRIPT" "${CFG[*]}" \
  hydra.run.dir="$EXP_ROOT_DIR" \
  +name="$EXPT_NAME" \
  +comment="$COMMENT" \
  +partition="$PARTITION" \
  +log_folder="$EXP_ROOT_DIR" \
  config.CHECKPOINT.DIR="$CHECKPOINT_DIR"
