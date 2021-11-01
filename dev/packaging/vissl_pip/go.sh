#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
rm -rf ClassyVision
git clone https://github.com/facebookresearch/ClassyVision.git

cd ClassyVision || exit
git reset --hard 4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d
cd ../ || exit
rm -rf ../../../classy_vision
cp -r ClassyVision/classy_vision ../../../classy_vision

rm -rf fairscale
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale || exit
git reset --hard df7db85cef7f9c30a5b821007754b96eb1f977b6
cd ../ || exit
rm -rf ../../../fairscale
cp -r fairscale/fairscale ../../../fairscale

docker run --rm  -v $PWD/../../..:/inside pytorch/conda-cuda bash inside/dev/packaging/vissl_pip/inside.sh
