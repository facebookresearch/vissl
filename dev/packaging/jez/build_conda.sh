#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

packaging=dev/packaging/jez

mkdir -p "$packaging/out"

version=$(python -c "exec(open('vissl/__init__.py').read()); print(__version__)")
build_version=$version.post$(date +%Y%m%d)

export BUILD_VERSION=$build_version

conda build -c bottler -c pytorch -c defaults --no-anaconda-upload --python "$PYTHON_VERSION" --output-folder "$packaging/out" "$packaging/vissl"
