#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -ex

conda init bash
# shellcheck source=/dev/null
source ~/.bashrc

conda create -y -n myenv python=3.7 opencv
conda activate myenv
conda install -y -c pytorch pytorch=1.5.1 cudatoolkit=10.1 torchvision

pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py37_cu101_pyt151/download.html apex
#pip install vissl --no-index -f https://dl.fbaipublicfiles.com/vissl/packaging/visslwheels/download.html
pip install vissl
python -c "import vissl, apex"
cd loc1
python -m unittest discover -v -s tests
dev/run_quick_tests.sh
