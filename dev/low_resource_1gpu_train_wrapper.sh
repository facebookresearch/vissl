#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

CFG=( "$@" )

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_DIR=$(dirname "${SRC_DIR}")
BINARY="python ${SRC_DIR}/tools/distributed_train.py"


echo "========================================================================"
echo "SRC_DIR: $SRC_DIR"
echo "Setting to run:"
echo "${CFG[@]}"
echo "========================================================================"


echo "Starting...."
# shellcheck disable=SC2102
# shellcheck disable=SC2086
$BINARY ${CFG[*]} \
    config.MACHINE.NUM_DATALOADER_WORKERS=0 \
    config.MACHINE.DEVICE=gpu \
    config.MULTI_PROCESSING_METHOD=forkserver \
    config.DISTRIBUTED.INIT_METHOD=tcp \
    config.DISTRIBUTED.RUN_ID=auto \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1
