#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SRC_DIR=$(dirname "${SRC_DIR}")


# -----------------------------------------------------------------------------
# Unit tests: running important unit tests in CI
# -----------------------------------------------------------------------------

TEST_LIST=(
    "test_dino_deit.py"
    "test_dino_deit_fsdp.py"
    "test_dino_xcit.py"
    "test_dino_xcit_fsdp.py"
    "test_extract_cluster.py"
    "test_extract_features.py"
    "test_finetuning.py"
    "test_ibot.py"
    "test_larc_fsdp.py"
    "test_layer_memory_tracking.py"
    "test_losses_gpu.py"
    "test_model_helpers.py"
    "test_regnet_fsdp.py"
    "test_regnet_fsdp_integration.py"
    "test_state_checkpoint_conversion.py"
    "test_state_checkpointing.py"
    "test_vit_fsdp.py"
)

echo "========================================================================"
echo "Unit tests to run:"
echo "${TEST_LIST[@]}"
echo "========================================================================"

pushd "${SRC_DIR}/tests"
for test_file in "${TEST_LIST[@]}"; do
  python -m unittest "$test_file" || exit
done
popd


# -----------------------------------------------------------------------------
# Integration tests: running configurations
# - verify that the configuration are valid
# - verify that the associated jobs run to the end
# -----------------------------------------------------------------------------

bash "${SRC_DIR}/dev/run_quick_integration_tests.sh"
