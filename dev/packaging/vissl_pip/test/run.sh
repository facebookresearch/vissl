#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

set -ex

#root=$PWD/../../../..
f(){
    echo -v $PWD/../../../../$1:/loc1/$1
}

sudo docker run --runtime=nvidia --shm-size 4000000000 -it --rm $(f dev) $(f configs) $(f tools) $(f tests) -v $PWD:/loc pytorch/conda-cuda bash /loc/test.sh
