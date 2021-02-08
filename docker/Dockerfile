ARG CUDA_VER
ARG CUDA_SUFFIX
ARG USER_ID
ARG TORCH_CUDA_ARCH_LIST

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

# for security concerns, create a non-root user
ARG USER_ID
RUN useradd -m --no-log-init --system --uid ${USER_ID} vissluser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER vissluser
WORKDIR /home/vissluser
ENV PATH="/home/vissluser/.local/bin:${PATH}"

# install PyTorch.
ARG CUDA_SUFFIX
RUN pip install --user torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/${CUDA_SUFFIX}/torch_stable.html

# install apex
ARG TORCH_CUDA_ARCH_LIST
ARG CUDA_VER
ADD ./common/install_apex.sh install_apex.sh
RUN bash ./install_apex.sh && rm install_apex.sh

# Install VISSL
RUN git clone https://github.com/facebookresearch/vissl vissl && cd vissl
RUN pip install --user -e vissl
WORKDIR /home/vissluser/vissl

CMD ["bash"]

######## run tests - cpu only
# python -m unittest discover -v -s tests

######## run gpu test
# python tools/run_distributed_engines.py \
#     config=test/circleci/quick_simclr_synthetic \
#     hydra.verbose=true config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true
