#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
rm -rf ClassyVision
git clone https://github.com/facebookresearch/ClassyVision.git
rm -rf ../../../classy_vision
cp -r ClassyVision/classy_vision ../../../classy_vision
rm -rf ../../../fairscale

sudo docker run --rm  -v $PWD/../../..:/inside pytorch/conda-cuda bash inside/dev/packaging/vissl_pip/inside.sh
