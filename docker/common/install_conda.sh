#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -ex

export TEMP_INSTALL=/tmp/conda-install
mkdir -p $TEMP_INSTALL

pushd $TEMP_INSTALL

# Anaconda
echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
sudo /bin/bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda

rm Miniconda3-latest-Linux-x86_64.sh

popd

# add conda to path
export PATH=/opt/conda/bin:$PATH

## Follow the steps below to create and activate a conda environment.
conda create --name vissl_env python=3.6
bash -c "source activate vissl_env"
export PATH="/opt/conda/envs/vissl_env/bin:${PATH}"

# some quick version checks
which conda
conda --version
which python
python --version
