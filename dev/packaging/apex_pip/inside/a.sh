#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
set -ex

conda init bash
# shellcheck source=/dev/null
source ~/.bashrc

cd /inside/apex
VERSION=0.0

export BUILD_VERSION=$VERSION

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Install with pip a bit more robustly than the default
pip_install() {
  retry pip install --progress-bar off "$@"
}


PYTHON_VERSIONS="3.6 3.7 3.8 3.9"
# the keys are pytorch versions
declare -A CONDA_CUDA_VERSIONS=(
    ["1.4.0"]="cu101"
    ["1.5.0"]="cu101 cu102"
    ["1.5.1"]="cu101 cu102"
    ["1.6.0"]="cu101 cu102"
    ["1.7.0"]="cu101 cu102 cu110"
    ["1.7.1"]="cu101 cu102 cu110"
    ["1.8.0"]="cu101 cu102 cu111"
    ["1.8.1"]="cu101 cu102 cu111"
    ["1.9.0"]="cu102 cu111"
    ["1.9.1"]="cu102 cu111"
)

#VERSION=$(python -c "exec(open('${script_dir}/apex/__init__.py').read()); print(__version__)")

for python_version in $PYTHON_VERSIONS
do
    for pytorch_version in "${!CONDA_CUDA_VERSIONS[@]}"
    do
        if [[ "3.6 3.7 3.8" != *$python_version* ]] && [[ "1.4.0 1.5.0 1.5.1 1.6.0 1.7.0" == *$pytorch_version* ]]
        then
            #python 3.9 and later not supported by pytorch 1.7.0 and before
            continue
        fi

        for cu_version in ${CONDA_CUDA_VERSIONS[$pytorch_version]}
        do
            extra_channel=""
            case "$cu_version" in
                cu111)
                    export CUDA_HOME=/usr/local/cuda-11.1/
                    export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0"
                    export CUDA_TAG=11.1
                    extra_channel="-c conda-forge"
                ;;
                cu110)
                    export CUDA_HOME=/usr/local/cuda-11.0/
                    export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0"
                    export CUDA_TAG=11.0
                ;;
                cu102)
                    export CUDA_HOME=/usr/local/cuda-10.2/
                    export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
                    export CUDA_TAG=10.2
                ;;
                cu101)
                    export CUDA_HOME=/usr/local/cuda-10.1/
                    export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
                    export CUDA_TAG=10.1
                ;;
                *)
                    echo "Unrecognized cu_version=$cu_version"
                    exit 1
                ;;
            esac

            if [[ "3.9" == "$python_version" ]]
            then
                extra_channel="-c conda-forge"
            fi

            tag=py"${python_version//./}"_"${cu_version}"_pyt"${pytorch_version//./}"

            outdir="../output/$tag"
            if [[ -d "$outdir" ]]
            then
                echo "skipping" "$outdir"
                continue
            fi

            conda create -y -n "$tag" "python=$python_version"
            conda activate "$tag"
            conda install -y -c pytorch $extra_channel "pytorch=$pytorch_version" "cudatoolkit=$CUDA_TAG"
            echo "python version" "$python_version" "pytorch version" "$pytorch_version" "cuda version" "$cu_version" "tag" "$tag"

            rm -rf dist

            python setup.py clean --cpp_ext --cuda_ext
            python setup.py bdist_wheel --cpp_ext --cuda_ext

            rm -rf "$outdir"
            mkdir -p "$outdir"
            cp dist/*whl "$outdir"
            #break
        done
    done
done
echo "DONE"
