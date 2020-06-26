#!/bin/bash

set -ex

# cu$CU_VERSION          # non-conda
# cu$CU_VERSION-conda    # conda
# Examples:
#   -> cu101 ./build_docker.sh
#   -> cu101-conda  ./build_docker.sh
image=${image}
if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

CUDA_VERSION="$(echo "${image}" | perl -n -e'/cu(\d+(?:\.\d+)?)/ && print $1')"
CONDA_ENV=0
USER_ID=${USER_ID-1000}
DOCKERFILE="./Dockerfile"
CUDA_SUFFIX="cu${CUDA_VERSION}"
IMAGE_TAG="vissl:1.0-${CUDA_SUFFIX}"

# Get setting whether to use conda or not
if [[ "$image" == *-conda* ]]; then
    CONDA_ENV=1
    DOCKERFILE="conda/Dockerfile"
    IMAGE_TAG="vissl:1.0-${CUDA_SUFFIX}-conda"
fi


# cuda versions
case "$CUDA_VERSION" in
  92)
    CUDA_VERSION=9.2
    TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX"
    ;;
  100)
    CUDA_VERSION=10.0
    TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX"
    ;;
  101)
    CUDA_VERSION=10.1
    TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX"
    ;;
  102)
    CUDA_VERSION=10.2
    TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX"
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=$CUDA_VERSION"
    exit 1
    ;;
esac


echo "============Printing summary============="
echo "image: ${image}"
echo "CUDA_SUFFIX: ${CUDA_SUFFIX}"
echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "PYTORCH_VERSION: ${PYTORCH_VERSION}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "USER_ID: ${USER_ID}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "IMAGE_TAG: ${IMAGE_TAG}"
echo "============Summary Ended================"


# Build image
docker build \
       --build-arg "BUILD_ENVIRONMENT=${image}" \
       --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
       --build-arg "CUDA_SUFFIX=${CUDA_SUFFIX}" \
       --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
       --build-arg "USER_ID=${USER_ID}" \
       -t ${IMAGE_TAG} \
       -f ${DOCKERFILE} \
       .
