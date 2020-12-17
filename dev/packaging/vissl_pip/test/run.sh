#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

set -ex

#root=$PWD/../../../..
f(){
    echo -v $(realpath $PWD/../../../../$1):/loc1/$1
}

sudo docker run --runtime=nvidia -it --rm $(f dev) $(f configs) $(f tools) $(f tests) -v $PWD:/loc pytorch/conda-cuda bash /loc/test.sh
