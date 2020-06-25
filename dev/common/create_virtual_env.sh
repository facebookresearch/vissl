#!/bin/bash

set -ex

if [ "$CONDA_ENV" == "1" ]; then
    conda create --name vissl_env python="$PYTHON_VERSION"
    echo "source activate vissl_env" > ~/.bashrc
    export PATH=/opt/conda/envs/vissl_env/bin:$PATH
else
    case "$PYTHON_VERSION" in
        3.6) python_abi=cp36-cp36m ;;
        3.7) python_abi=cp37-cp37m ;;
        3.8) python_abi=cp38-cp38 ;;
        *)
        echo "Unrecognized PYTHON_VERSION=$PYTHON_VERSION"
        exit 1
        ;;
    esac
    export PATH="/opt/python/$python_abi/bin:$PATH"

    python -m venv /opt/vissl_venv
    export PATH="/opt/vissl_venv/bin:$PATH"
fi
