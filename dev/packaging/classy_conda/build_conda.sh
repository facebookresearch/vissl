#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

mkdir -p "./out"

conda build -c pytorch -c conda-forge -c defaults --no-anaconda-upload --python "$PYTHON_VERSION" --output-folder "./out" "./ClassyVision"
