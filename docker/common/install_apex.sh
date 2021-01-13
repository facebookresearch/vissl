#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

set -ex

export TEMP_INSTALL=/tmp/apex-install
mkdir -p $TEMP_INSTALL

pushd $TEMP_INSTALL

# set the cuda paths and check for the nvidia-compiler version
export PATH=/usr/local/cuda-${CUDA_VER}/bin:/usr/local/bin:$PATH
# now, check the nvcc is available. the following command should print nvcc path
nvcc --version

# see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
# maxwell, pascal, volta
# example: TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.5"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# install apex now (note that we recommend a specific apex version for stability)
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex@https://github.com/NVIDIA/apex/tarball/1f2aa9156547377a023932a1512752c392d9bbdf

popd

# Verify apex installed
python -c 'import apex'
