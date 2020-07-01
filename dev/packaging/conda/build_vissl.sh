#!/usr/bin/env bash
set -ex

# cu$CU_VERSIONpyt$PYTORCH_VERSION
# Examples:
#   -> cu101-pyt1.5 ./package_vissl.sh
image=${image}
if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
VISSL_SOURCE_DIR="$(readlink -f -- "$(dirname -- "$script_dir")/../..")"
CUDA_VER="$(echo "${image}" | perl -n -e'/cu(\d+(?:\.\d+)?)/ && print $1')"
PYTORCH_VERSION="$(echo "${image}" | perl -n -e'/-pyt(\d+(?:\.\d+)?)/ && print $1')"
BUILD_VERSION=$(python -c "exec(open('$VISSL_SOURCE_DIR/vissl/__init__.py').read()); print(__version__)")
BUILD_NUMBER=1
PYTHON_VERSIONS=('3.6' '3.7' '3.8')
CUDA_SUFFIX="cu${CUDA_VER}"
VISSL_FINAL_PACKAGE_DIR="/tmp/.X11-unix"

# cuda versions
case "$CUDA_VER" in
  92)
    CUDA_VER=9.2
    export CONDA_CUDATOOLKIT_VERSION="- cudatoolkit >=9.2,<9.3 # [not osx]"
    ;;
  100)
    CUDA_VER=10.0
    export CONDA_CUDATOOLKIT_VERSION="- cudatoolkit >=10.0,<10.1 # [not osx]"
    ;;
  101)
    CUDA_VER=10.1
    export CONDA_CUDATOOLKIT_VERSION="- cudatoolkit >=10.1,<10.2 # [not osx]"
    ;;
  102)
    CUDA_VER=10.2
    export CONDA_CUDATOOLKIT_VERSION="- cudatoolkit=10.2 # [not osx]"
    ;;
  *)
    echo "Unrecognized CUDA_VER=$CUDA_VER"
    exit 1
    ;;
esac

export CONDA_PYTORCH_VERSION="- pytorch==$PYTORCH_VERSION"
export PYTORCH_VERSION_NODOT=${PYTORCH_VERSION//./}
export BUILD_VERSION=$BUILD_VERSION
export BUILD_NUMBER=$BUILD_NUMBER
export CUDA_VER=$CUDA_VER
export CUDA_SUFFIX=$CUDA_SUFFIX
export VISSL_SOURCE_DIR=$VISSL_SOURCE_DIR

# set the anaconda upload to NO for now
conda config --set anaconda_upload no
conda config --add channels conda-forge
conda config --add channels pytorch
# install conda-build
conda install -yq conda-build

echo "Packaging VISSL ==> BUILD_VERSION: ${BUILD_VERSION} BUILD_NUMBER: ${BUILD_NUMBER}"
# Loop through all Python versions to build a package for each
for py_ver in "${PYTHON_VERSIONS[@]}"; do
    # for cuda 9.2 and python 3.6, we only support pytorch1.4
    if [[ "$CUDA_VER" = "9.2" &&  "$py_ver" = "3.6" && "$PYTORCH_VERSION" = "1.5" ]]; then
      echo "For cuda9.2 and python3.6, only pytorch1.4 is supported"
      continue
    fi
    export PYTHON_NODOT=${py_ver//./}
    build_string="py${PYTHON_NODOT}_${CUDA_SUFFIX}_pytorch${PYTORCH_VERSION_NODOT}"
    folder_tag="${build_string}_$(date +'%Y%m%d')"

    # Create the conda package into this temporary folder. This is so we can find
    # the package afterwards, as there's no easy way to extract the final filename
    # from conda-build
    output_folder="out_$folder_tag"
    rm -rf "$output_folder"
    mkdir "$output_folder"
    echo $output_folder

    # if the built package already exists, we skip it
    if [[ -n "$VISSL_FINAL_PACKAGE_DIR" ]]; then
        built_package="$(find $VISSL_FINAL_PACKAGE_DIR/ -name *${build_string}*.tar.bz2)"
        if [[ -n "$built_package" ]]; then
            echo "found package: $built_package"
            continue
        fi
    fi

    echo "Calling conda-build at $(date)"
    time conda build -c conda-forge \
                     -c pytorch \
                     --no-anaconda-upload \
                     --python "$py_ver" \
                     --output-folder "$output_folder" \
                     --keep-old-work \
                     --no-remove-work-dir \
                     vissl
    echo "Finished conda-build at $(date)"

    # Extract the package for testing
    ls -lah "$output_folder"
    built_package="$(find $output_folder/ -name '*vissl*.tar.bz2')"

    # Copy the built package to the host machine for persistence before testing
    if [[ -n "$VISSL_FINAL_PACKAGE_DIR" ]]; then
        mkdir -p "$VISSL_FINAL_PACKAGE_DIR" || true
        cp "$built_package" "$VISSL_FINAL_PACKAGE_DIR/"
    fi
done
