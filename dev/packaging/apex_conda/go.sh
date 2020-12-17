#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
sudo docker run --rm  -v $PWD/inside:/inside pytorch/conda-cuda python3 inside/build.py
