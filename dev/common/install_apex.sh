#!/bin/bash

set -ex

export TEMP_INSTALL=/tmp/apex-install
mkdir -p $TEMP_INSTALL

pushd $TEMP_INSTALL

# set the nvidia compiler
module load cuda/10.1
# now, check the nvcc is available. the following command should print nvcc path
which nvcc

# install the correct version of gcc between gcc 7.1 and 7.3
# cleanup again to avoid any sha mismatch
apt-get clean
rm -rf /var/lib/apt/lists/*
# setup gcc
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y software-properties-common build-essential g++-multilib
apt-get upgrade -y
apt-get install -y gcc-7-multilib g++-7-multilib
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7

export TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# install apex now (note that we recommend a specific apex version for stability)
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex@https://github.com/NVIDIA/apex/tarball/1f2aa9156547377a023932a1512752c392d9bbdf

# Verify apex installed
python -c 'import apex'


# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
