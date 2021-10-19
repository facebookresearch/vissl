#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_DIR=$(dirname "${SRC_DIR}")

# -----------------------------------------------------------------------------
# Integration tests: running configurations
# - verify that the configuration are valid
# - verify that the associated jobs run to the end
# -----------------------------------------------------------------------------

CFG_LIST=(
    "test/integration_test/quick_barlow_twins"
    "test/integration_test/quick_deepcluster_v2"
    "test/integration_test/quick_pirl"
    "test/integration_test/quick_simclr"
    "test/integration_test/quick_simclr_efficientnet"
    "test/integration_test/quick_simclr_multicrop"
    "test/integration_test/quick_simclr_regnet"
    "test/integration_test/quick_swav"
)

echo "========================================================================"
echo "Configs to run:"
echo "${CFG_LIST[@]}"
echo "========================================================================"

BINARY="python ${SRC_DIR}/tools/run_distributed_engines.py"

for cfg in "${CFG_LIST[@]}"; do
    echo "========================================================================"
    echo "Running $cfg ..."
    echo "========================================================================"
    CHECKPOINT_DIR=$(mktemp -d)
    # shellcheck disable=SC2102
    # shellcheck disable=SC2086
    CUDA_LAUNCH_BLOCKING=1 $BINARY config=$cfg \
        config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
        hydra.verbose=true \
        config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
        config.CHECKPOINT.DIR="$CHECKPOINT_DIR" && echo "TEST OK" || exit

    rm -rf "$CHECKPOINT_DIR"
done
