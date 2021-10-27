#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -ex

rm -rf dev/packaging/vissl_conda/ClassyVision
git clone https://github.com/facebookresearch/ClassyVision.git dev/packaging/vissl_conda/ClassyVision
cd dev/packaging/vissl_conda/ClassyVision
git reset --hard 4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d
cd ../../../../

rm -rf dev/packaging/vissl_conda/fairscale
git clone https://github.com/facebookresearch/fairscale.git dev/packaging/vissl_conda/fairscale
cd dev/packaging/vissl_conda/fairscale
git reset --hard df7db85cef7f9c30a5b821007754b96eb1f977b6
cd ../../../../

rm -rf classy_vision
cp -r dev/packaging/vissl_conda/ClassyVision/classy_vision classy_vision
rm -rf fairscale
cp -r dev/packaging/vissl_conda/fairscale/fairscale fairscale

for PV in 3.6 3.7 3.8
do
   PYTHON_VERSION=$PV bash dev/packaging/vissl_conda/build_conda.sh
done
