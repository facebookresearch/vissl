#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export SOURCE_ROOT_DIR="/inside/apex"
export CONDA_CPUONLY_FEATURE=

PYTHON_VERSIONS="3.6 3.7 3.8"
# the keys are pytorch versions
declare -A CONDA_CUDA_VERSIONS=(
    ["1.4.0"]="cu101"
    ["1.5.0"]="cu101 cu102"
    ["1.5.1"]="cu101 cu102"
    ["1.6.0"]="cu101 cu102"
    ["1.7.0"]="cu101 cu102 cu110"
)
# for ptv in "${!CONDA_CUDA_VERSIONS[@]}"; do echo "$ptv - ${CONDA_CUDA_VERSIONS[$ptv]}"; done

VERSION=0.0
# Uncomment this to use the official version number
#VERSION=$(python -c "exec(open('${script_dir}/apex/amp/__init__.py').read()); print(__version__)")
export BUILD_VERSION=$VERSION

for python_version in $PYTHON_VERSIONS
do
    export PYTHON_VERSION=${python_version}
    for ptv in ${!CONDA_CUDA_VERSIONS[@]}
    do
        export PYTORCH_VERSION=${ptv}
        for cuv in ${CONDA_CUDA_VERSIONS[$ptv]}
        do
            export CU_VERSION=${cuv}
            export PYTORCH_VERSION_NODOT=${PYTORCH_VERSION//./}
            export CONDA_PYTORCH_BUILD_CONSTRAINT="- pytorch==${PYTORCH_VERSION}"
            export CONDA_PYTORCH_CONSTRAINT="- pytorch==${PYTORCH_VERSION}"


            case "$CU_VERSION" in
                cu110)
                    export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=11.0,<11.1 # [not osx]"
                    export CUDA_HOME=/usr/local/cuda-11.0/
                ;;
                cu102)
                    export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.2,<10.3 # [not osx]"
                    export CUDA_HOME=/usr/local/cuda-10.2/
                ;;
                cu101)
                    export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.1,<10.2 # [not osx]"
                    export CUDA_HOME=/usr/local/cuda-10.1/
                ;;
                cu100)
                    export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=10.0,<10.1 # [not osx]"
                    export CUDA_HOME=/usr/local/cuda-10.0/
                ;;
                cu92)
                    export CONDA_CUDATOOLKIT_CONSTRAINT="- cudatoolkit >=9.2,<9.3 # [not osx]"
                    export CUDA_HOME=/usr/local/cuda-9.2/
                ;;
                cpu)
                    export CONDA_CUDATOOLKIT_CONSTRAINT=""
                    export CONDA_CPUONLY_FEATURE="- cpuonly"
                    #need to do something for export CUDA_HOME=/usr/local/cuda-10.2/?
                ;;
                *)
                    echo "Unrecognized CU_VERSION=$CU_VERSION"
                    exit 1
                ;;
            esac
            echo "python version" "$PYTHON_VERSION" "pytorch version" "$PYTORCH_VERSION" "cuda version" "$CU_VERSION"
            conda build -c pytorch -c defaults --no-anaconda-upload --python "$PYTHON_VERSION" inside/packaging/apex
            cp -r /opt/conda/conda-bld/linux-64/apex-"${VERSION}"-py"${PYTHON_VERSION//./}"_"${CU_VERSION}"_pyt"${PYTORCH_VERSION_NODOT}".tar.bz2 inside/packaging
        done
    done
done
echo "DONE"
