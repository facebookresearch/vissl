ARG CUDA_VERSION
ARG CUDA_SUFFIX
ARG USER_ID
ARG PYTORCH_VERSION

# we always use cudnn version 7
FROM nvidia/cuda:${CUDA_VERSION}-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends ca-certificates wget sudo git vim
RUN rm -rf /var/lib/apt/lists/*

# for security concerns, create a non-root user
RUN useradd -m --no-log-init --system --uid ${USER_ID} vissluser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER vissluser
WORKDIR /home/vissluser
ENV PATH="/home/vissluser/.local/bin:${PATH}"

# pip
ADD ../dev/common/install_pip_python.sh install_pip_python.sh
RUN bash ./install_pip_python.sh && rm install_pip_python.sh

# Anaconda
ADD ../dev/common/install_conda.sh install_conda.sh
RUN if [ "CONDA_ENV" == "1" ]; then bash ./install_conda.sh && rm install_conda.sh; fi

# Virtual environment
ARG CONDA_ENV
ARG PYTHON_VERSION
ADD ../dev/common/create_virtual_env.sh create_virtual_env.sh
RUN bash ./create_virtual_env.sh && rm create_virtual_env.sh

# install PyTorch.
# use --no-index to avoid mistakenly downloading wrong cuda version from pypi
RUN pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 --no-index -f https://download.pytorch.org/whl/torch_stable.html

# install apex
ARG TORCH_CUDA_ARCH_LIST
ADD ../dev/common/install_apex.sh install_apex.sh
RUN bash ./install_apex.sh && rm install_apex.sh

# Install VISSL
RUN git clone https://github.com/facebookresearch/vissl vissl
RUN pip install --user -e vissl
WORKDIR /home/vissluser/vissl

CMD ["bash"]
