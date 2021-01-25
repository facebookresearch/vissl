#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
rm -rf ClassyVision
git clone https://github.com/facebookresearch/ClassyVision.git
rm -rf ../../../classy_vision
cp -r ClassyVision/classy_vision ../../../classy_vision
rm -rf ../../../fairscale

sudo docker run --rm  -v $PWD/../../..:/inside pytorch/conda-cuda bash inside/dev/packaging/vissl_pip/inside.sh
