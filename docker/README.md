# How to build VISSL docker images

VISSL supports (10.2, 11.1) CUDA version for these docker containers for both the pip and conda environments.
We provide a parameterized `Dockerfile` to build the image based on the build environment. The different configurations are identified by a freeform string that we call a build environment. We support latest pytorch version (1.9.1) in our docker container. If you wish to change the pytorch version, please modify the Dockerfile pytorch installation commands.

## Build environments

You can specify build environment with string:
- `cu$CU_VERSION`          # pip
- `cu$CU_VERSION-conda`    # conda

Examples:
- Pip environment: `cu102`
- Conda environment: `c102-conda`

See `build_docker.sh` for a full list of terms that are extracted from the build environment into parameters for the image build.


## Steps to build docker image

**NOTE**: You need to have docker installed on your system. Follow the instructions
on docker website to install it, if you don't have docker already.

You can verify your docker installation is fine by running a docker test:

```bash
docker run hello-world
```

1. Clone VISSL repo

```bash
cd $HOME && git clone --recursive git@github.com:facebookresearch/vissl.git && cd $HOME/vissl/
```

2. Build the docker image

- For pip environment

```bash
image=cu101 ./build_docker.sh
```

- For conda environment
```bash
image=cu101-conda ./build_docker.sh
```

This will build the image for the above permutation and then we can test this image

6. Test the image

run `docker images` and get the `IMAGE_ID` for the image we just built.

```bash
docker run --gpus all -it \
    --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" ${IMAGE_ID}
```
