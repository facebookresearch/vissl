#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
sudo docker run --rm  -v $PWD/inside:/inside pytorch/conda-cuda bash inside/a.sh
