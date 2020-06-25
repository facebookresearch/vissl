#!/bin/bash

set -ex

export TEMP_INSTALL=/tmp/conda-install
mkdir -p $TEMP_INSTALL

pushd $TEMP_INSTALL

# Anaconda
echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x  miniconda.sh
/bin/bash .miniconda.sh -b -p /opt/conda

rm miniconda.sh

popd

# add conda to path
export PATH=/opt/conda/bin:$PATH

# install dependencies
conda install -y conda-build anaconda-client git ninja

# some quick version checks
which conda
conda --version
which python
python --version

# Cleanup package manager
apt-get autoclean && apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
