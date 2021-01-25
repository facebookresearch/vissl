
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

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
!pip install vissl

import vissl
import tensorboard
import apex
import torch

!mkdir -p configs/config
!wget -q -O configs/__init__.py https://dl.fbaipublicfiles.com/vissl/tutorials/configs/__init__.py 
!wget -q -O configs/config/eval_resnet_8gpu_transfer_in1k_fulltune.yaml https://dl.fbaipublicfiles.com/vissl/tutorials/configs/eval_resnet_8gpu_transfer_in1k_fulltune.yaml

!wget https://download.pytorch.org/models/resnet50-19c8e357.pth

!wget https://dl.fbaipublicfiles.com/vissl/tutorials/run_distributed_engines.py

!mkdir -p dummy_data/train/class1
!mkdir -p dummy_data/train/class2
!mkdir -p dummy_data/val/class1
!mkdir -p dummy_data/val/class2

# create 2 classes in train and add 5 images per class
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class1/img1.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class1/img2.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class1/img3.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class1/img4.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class1/img5.jpg

!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class2/img1.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class2/img2.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class2/img3.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class2/img4.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/train/class2/img5.jpg

# create 2 classes in val and add 5 images per class
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class1/img1.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class1/img2.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class1/img3.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class1/img4.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class1/img5.jpg

!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class2/img1.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class2/img2.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class2/img3.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class2/img4.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O dummy_data/val/class2/img5.jpg


json_data = {
    "dummy_data_folder": {
      "train": [
        "/content/dummy_data/train", "/content/dummy_data/train"
      ],
      "val": [
        "/content/dummy_data/val", "/content/dummy_data/val"
      ]
    }
}

# use VISSL's api to save or you can use your custom code.
from vissl.utils.io import save_file
save_file(json_data, "/content/configs/config/dataset_catalog.json")

from vissl.data.dataset_catalog import VisslDatasetCatalog

# list all the datasets that exist in catalog
print(VisslDatasetCatalog.list())

# get the metadata of dummy_data_folder dataset
print(VisslDatasetCatalog.get("dummy_data_folder"))

!python3 run_distributed_engines.py \
    hydra.verbose=true \
    config=eval_resnet_8gpu_transfer_in1k_fulltune \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATASET_NAMES=[dummy_data_folder] \
    config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=2 \
    config.DATA.TEST.DATA_SOURCES=[disk_folder] \
    config.DATA.TEST.LABEL_SOURCES=[disk_folder] \
    config.DATA.TEST.DATASET_NAMES=[dummy_data_folder] \
    config.DATA.TEST.BATCHSIZE_PER_REPLICA=2 \
    config.OPTIMIZER.num_epochs=2 \
    config.OPTIMIZER.param_schedulers.lr.values=[0.01,0.001] \
    config.OPTIMIZER.param_schedulers.lr.milestones=[1] \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
    config.CHECKPOINT.DIR="./checkpoints" \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/content/resnet50-19c8e357.pth" \
    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=""


ls checkpoints/

cat checkpoints/metrics.json
