#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_DIR=$(dirname "${SRC_DIR}")
BINARY="python ${SRC_DIR}/tools/distributed_train.py"
CONFIG_PATH="${SRC_DIR}/hydra_configs/"

CFG_LIST=(
    "test/integration_test/quick_simclr"
    "test/integration_test/quick_simclr_multicrop"
    "test/integration_test/quick_pirl"
    "test/integration_test/quick_simclr_efficientnet"
    "test/integration_test/quick_swav"
    "test/integration_test/quick_deepcluster_v2"
)

echo "========================================================================"
echo "Configs to run:"
echo "${CFG_LIST[@]}"
echo "========================================================================"

for cfg in "${CFG_LIST[@]}"; do
    echo "========================================================================"
    echo "Running $cfg ..."
    echo "========================================================================"
    # shellcheck disable=SC2102
    # shellcheck disable=SC2086
    $BINARY --config-path=$CONFIG_PATH config=$cfg \
        config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
        hydra.verbose=true \
        config.TENSORBOARD_SETUP.USE_TENSORBOARD=true
done
