#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# cu$CU_VERSION          # non-conda
# cu$CU_VERSION-conda    # conda
# Examples:
#   -> image=cu101 ./build_docker.sh
#   -> image=cu101-conda  ./build_docker.sh

image=${image}
if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

CUDA_VER="$(echo "${image}" | perl -n -e'/cu(\d+(?:\.\d+)?)/ && print $1')"
CONDA_ENV=0
USER_ID=${USER_ID-1000}
DOCKERFILE="./Dockerfile"
CUDA_SUFFIX="cu${CUDA_VER}"
IMAGE_TAG="vissl:1.0-${CUDA_SUFFIX}"
 # You can choose a specific VISSL branch or commit to run. e.g. main or v0.1.6.
VISSL_BRANCH=${VISSL_BRANCH-v0.1.6}
APEX_CUDA_SUFFIX="${CUDA_SUFFIX//./}"
# Get setting whether to use conda or not
if [[ "$image" == *-conda* ]]; then
    CONDA_ENV=1
    DOCKERFILE="conda/Dockerfile"
    IMAGE_TAG="vissl:1.0-${CUDA_SUFFIX}-conda"
fi


# cuda versions
case "$CUDA_VER" in
  101)
    CUDA_VER=10.1
    TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX"
    ;;
  102)
    CUDA_VER=10.2
    TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX"
    ;;
  *)
    echo "Unrecognized CUDA_VER=$CUDA_VER"
    exit 1
    ;;
esac


echo "============Printing summary============="
echo "image: ${image}"
echo "CUDA_SUFFIX: ${CUDA_SUFFIX}"
echo "CUDA_VER: ${CUDA_VER}"
echo "APEX_CUDA_SUFFIX: ${APEX_CUDA_SUFFIX}"
echo "VISSL_BRANCH: ${VISSL_BRANCH}"
echo "PYTORCH_VERSION: ${PYTORCH_VERSION}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "USER_ID: ${USER_ID}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "IMAGE_TAG: ${IMAGE_TAG}"
echo "============Summary Ended================"


# Build image
# shellcheck disable=SC2102
# shellcheck disable=SC2086
docker build \
       --build-arg "BUILD_ENVIRONMENT=${image}" \
       --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
       --build-arg "CUDA_SUFFIX=${CUDA_SUFFIX}" \
       --build-arg "CUDA_VER=${CUDA_VER}" \
       --build-arg "USER_ID=${USER_ID}" \
       --build-arg "VISSL_BRANCH=${VISSL_BRANCH}" \
       --build-arg "APEX_CUDA_SUFFIX=${APEX_CUDA_SUFFIX}" \
       -t ${IMAGE_TAG} \
       -f ${DOCKERFILE} \
       --progress=plain \
       .
