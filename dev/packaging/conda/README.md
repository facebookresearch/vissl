# How to build VISSL conda packages

## Step 1: Setup small docker image

We build conda packages for VISSL inside docker as this helps isolate the environment.

```bash
cd $HOME/vissl/dev/packaging/conda
# build docker (change the cuda version to desired version)
docker build --build-arg CUDA_VER=10.1 -t vissl_conda:cu101 -f Dockerfile .
```

## Step 2: Package VISSL

```bash
# run docker with gpus access
docker run --gpus all -it --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" vissl_conda:cu101
# clone vissl
git clone https://github.com/facebookresearch/vissl vissl && cd vissl/dev/packaging/conda
# conda package vissl
image=cu101-pyt1.5 ./build_vissl.sh
```

**NOTE**: If the packaging fails, you can cleanup the broken packages by running following command:
```bash
conda build purge
```

# Packages

CUDA |   torch1.5  |  torch1.4  |            python
-------------------------------------------------------------
10.2 |    yes      |            |  3.6, 3.7, 3.8
10.1 |    yes      |   yes      |  3.6, 3.7, 3.8
10.0 |             |   yes      |  3.6, 3.7, 3.8
9.2  |    yes      |   yes      |  3.6 (pyt14 only), 3.7, 3.8
-------------------------------------------------------------
