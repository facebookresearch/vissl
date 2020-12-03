#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

conda init bash
# shellcheck source=/dev/null
source ~/.bashrc

cd /inside

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Install with pip a bit more robustly than the default
pip_install() {
  retry pip install --progress-bar off "$@"
}


PYTHON_VERSIONS="3.6 3.7 3.8"
# the keys are pytorch versions

for python_version in $PYTHON_VERSIONS
do
    outdir="dev/packaging/vissl_pip/output/$python_version"
    tag="py$python_version"
    conda create -y -n "$tag" "python=$python_version"
    conda activate "$tag"
    echo "python version" "$python_version" "tag" "$tag"

    rm -rf dist

    python setup.py clean
    python setup.py bdist_wheel

    rm -rf "$outdir"
    mkdir -p "$outdir"
    cp dist/*whl "$outdir"
done
echo "DONE"
