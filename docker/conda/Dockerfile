ARG CUDA_VER
ARG CUDA_SUFFIX

# we always use cudnn version 7
FROM nvidia/cuda:${CUDA_VER}-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive

# cuda paths
ENV PATH /usr/local/cuda-${CUDA_VER}/bin:/usr/local/bin:$PATH

# Install common dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends ca-certificates wget sudo git vim nano python3-opencv python3-dev python3-pip
RUN rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip
RUN pip install -U setuptools

# Anaconda
ADD ./common/install_conda.sh install_conda.sh
RUN ./install_conda.sh && rm install_conda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# install PyTorch.
RUN conda install pytorch==1.9.1 torchvision cudatoolkit=${CUDA_VER} -c pytorch -c conda-forge

# install apex
RUN conda install apex -c vissl

# install classyvision
RUN pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/main
# TODO: Set Classyvision commit.

# Install VISSL
ARG VISSL_BRANCH
RUN git clone https://github.com/facebookresearch/vissl vissl && cd vissl && git checkout ${VISSL_BRANCH} && git checkout -b ${VISSL_BRANCH}
RUN pip install --user -e vissl
WORKDIR /vissl

CMD ["bash"]

######## run tests - cpu only
# python -m unittest discover -v -s tests

######## run gpu test
# python tools/run_distributed_engines.py \
#     config=test/circleci/quick_simclr_synthetic \
#     hydra.verbose=true config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true
