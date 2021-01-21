
# Install: PyTorch (we assume 1.5.1 but VISSL works with all PyTorch versions >=1.4)
!pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install opencv
!pip install opencv-python

# install apex by checking system settings: cuda version, pytorch version, python version
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

# install VISSL
!pip install vissl -f https://dl.fbaipublicfiles.com/vissl/packaging/visslwheels/download.html

import vissl
import tensorboard
import apex
import torch

!mkdir -p configs/config/
!wget -O configs/__init__.py https://dl.fbaipublicfiles.com/vissl/tutorials/configs/__init__.py 
!wget -O configs/config/quick_1gpu_resnet50_simclr.yaml https://dl.fbaipublicfiles.com/vissl/tutorials/configs/quick_1gpu_resnet50_simclr.yaml

!wget https://dl.fbaipublicfiles.com/vissl/tutorials/run_distributed_engines.py

!python3 run_distributed_engines.py \
    hydra.verbose=true \
    config=quick_1gpu_resnet50_simclr \
    config.DATA.TRAIN.DATA_SOURCES=[synthetic] \
    config.CHECKPOINT.DIR="./checkpoints" \
    config.TENSORBOARD_SETUP.USE_TENSORBOARD=true

ls checkpoints/

# Look at training curves in tensorboard:
!kill 490
%reload_ext tensorboard
%tensorboard --logdir checkpoints/tb_logs
