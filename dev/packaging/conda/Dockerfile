ARG CUDA_VER

# we always use cudnn version 7
FROM nvidia/cuda:${CUDA_VER}-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive

# cuda paths
ARG CUDA_VER
ENV PATH /usr/local/cuda-${CUDA_VER}/bin:/usr/local/bin:$PATH

# Install common dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends ca-certificates wget sudo git vim nano python-opencv python3-dev python3-pip

# install conda
RUN echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH

RUN echo $PATH
RUN which conda
RUN conda --version
WORKDIR /root/
