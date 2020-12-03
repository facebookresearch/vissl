#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

for PV in 3.6 3.7 3.8
do
   PYTHON_VERSION=$PV bash dev/packaging/vissl_conda/build_conda.sh
done
