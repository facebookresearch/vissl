#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

BINARY="python tools/distributed_train.py"
CONFIG_PATH="$HOME/vissl/hydra_configs/"

CFG_LIST=(
    "test/integration_test/quick_simclr"
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
    $BINARY --config-path=$CONFIG_PATH config=$cfg \
        config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
        hydra.verbose=true \
        config.TENSORBOARD_SETUP.USE_TENSORBOARD=true
done
