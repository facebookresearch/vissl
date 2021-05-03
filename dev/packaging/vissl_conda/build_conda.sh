#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -ex

packaging=dev/packaging/vissl_conda

mkdir -p "$packaging/out"

version=$(python -c "exec(open('vissl/__init__.py').read()); print(__version__)")
build_version=$version

export BUILD_VERSION=$build_version

# We allow the vissl channel to get the apex package.
# We specify it with a full url to avoid a name clash with a local directory called vissl.
# Having defaults before conda-forge is so that the tensorboard used in the
# tests will work.
conda build -c https://conda.anaconda.org/vissl -c iopath -c pytorch -c defaults -c conda-forge --no-anaconda-upload --python "$PYTHON_VERSION" --output-folder "$packaging/out" "$packaging/vissl"
