#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

set -e

# code gets checked out at below:
# $SOURCE_DIR ===> <anaconda_root>/conda-bld/vissl_<timestamp>/work
#
# build directory gets created at $SOURCE_DIR/build
#
# CONDA environment for debugging:
# cd <anaconda_root>/conda-bld/vissl_<timestamp>
# source activate ./_h_env_......    # long placeholders
#
# $CONDA_PREFIX and $PREFIX are set to the same value i.e. the environment value
#
# Installation happens in the $PREFIX which is the environment and rpath is set
# to that
#
# For tests, a new environment _test_env_.... is created
# During the tests, you will see that the vissl package gets checked out

echo $VISSL_SOURCE_DIR

# install apex
export PATH=/usr/local/cuda-${CUDA_VER}/bin:/usr/local/bin:$PATH
# now, check the nvcc is available. the following command should print nvcc path
nvcc --version
# see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
# maxwell, pascal, volta
export TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.5"
if [[ "$CUDA_VER" = "9.2" ]]; then
    export TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0"
fi

# install apex now (note that we recommend a specific apex version for stability)
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex@https://github.com/NVIDIA/apex/tarball/1f2aa9156547377a023932a1512752c392d9bbdf
# install hydra and classy vision (we pin to specific versions for constant tracking)
pip install hydra-core@https://github.com/facebookresearch/hydra/tarball/1.0_branch
pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/master

# install vissl
python setup.py install --single-version-externally-managed --record=record.txt

# Verify installs
python -c 'import apex'
python -c 'import classy_vision'
python -c 'import hydra'

# list what's installed in the environment
conda list
pip list
