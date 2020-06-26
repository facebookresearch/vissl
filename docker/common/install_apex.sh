#!/bin/bash

set -ex

export TEMP_INSTALL=/tmp/apex-install
mkdir -p $TEMP_INSTALL

pushd $TEMP_INSTALL

# set the cuda paths and check for the nvidia-compiler version
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:$PATH
# now, check the nvcc is available. the following command should print nvcc path
which nvcc

export TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# install apex now (note that we recommend a specific apex version for stability)
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex@https://github.com/NVIDIA/apex/tarball/1f2aa9156547377a023932a1512752c392d9bbdf

# Verify apex installed
python -c 'import apex'
