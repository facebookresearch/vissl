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
conda install pytorch==1.9.1 torchvision cudatoolkit=10.2 -c pytorch -c conda-forge

pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py37_cu102_pyt191/download.html apex
pip install vissl -f https://dl.fbaipublicfiles.com/vissl/packaging/visslwheels/download.html
pip install augly
python -c "import vissl, apex, augly"
cd loc1
python -m unittest discover -v -s tests
dev/run_quick_tests.sh
