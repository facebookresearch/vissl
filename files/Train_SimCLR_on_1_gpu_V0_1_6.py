
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Install pytorch version 1.8
!pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install Apex by checking system settings: cuda version, pytorch version, and python version
import sys
import torch
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{torch.__version__[0:5:2]}"
])
print(version_str)

# install apex (pre-compiled with optimizer C++ extensions and CUDA kernels)
!pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/{version_str}/download.html

# # clone vissl repository and checkout latest version.
!git clone --recursive https://github.com/facebookresearch/vissl.git

%cd vissl/

!git checkout v0.1.6
!git checkout -b v0.1.6

# install vissl dependencies
!pip install --progress-bar off -r requirements.txt
!pip install opencv-python

# update classy vision install to commit compatible with v0.1.6
!pip uninstall -y classy_vision
!pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d

# install vissl dev mode (e stands for editable)
!pip install -e .[dev]

import vissl
import tensorboard
import apex
import torch

!python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config=test/integration_test/quick_simclr.yaml \
    config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
    config.CHECKPOINT.DIR="/content/checkpoints" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true

ls /content/checkpoints/

# Look at training curves in tensorboard:
%reload_ext tensorboard
%tensorboard --logdir /content/checkpoints/tb_logs
