#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -ex

#root=$PWD/../../../..
f(){
    echo -v $PWD/../../../../$1:/loc1/$1
}

sudo docker run --runtime=nvidia --shm-size 4000000000 -it --rm $(f dev) $(f configs) $(f tools) $(f tests) -v $PWD:/loc pytorch/conda-cuda bash /loc/test.sh
