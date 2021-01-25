#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

rm -rf dev/packaging/vissl_conda/ClassyVision
git clone https://github.com/facebookresearch/ClassyVision.git dev/packaging/vissl_conda/ClassyVision
rm -rf dev/packaging/vissl_conda/fairscale
git clone https://github.com/facebookresearch/fairscale.git dev/packaging/vissl_conda/fairscale
rm -rf classy_vision
cp -r dev/packaging/vissl_conda/ClassyVision/classy_vision classy_vision
rm -rf fairscale
cp -r dev/packaging/vissl_conda/fairscale/fairscale fairscale

for PV in 3.6 3.7 3.8
do
   PYTHON_VERSION=$PV bash dev/packaging/vissl_conda/build_conda.sh
done
