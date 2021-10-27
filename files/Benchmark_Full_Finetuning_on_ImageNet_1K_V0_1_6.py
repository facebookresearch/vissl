
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

!wget https://download.pytorch.org/models/resnet50-19c8e357.pth -P /content/

!mkdir -p /content/dummy_data/train/class1
!mkdir -p /content/dummy_data/train/class2
!mkdir -p /content/dummy_data/val/class1
!mkdir -p /content/dummy_data/val/class2

# create 2 classes in train and add 5 images per class
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class1/img1.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class1/img2.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class1/img3.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class1/img4.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class1/img5.jpg

!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class2/img1.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class2/img2.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class2/img3.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class2/img4.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/train/class2/img5.jpg

# create 2 classes in val and add 5 images per class
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class1/img1.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class1/img2.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class1/img3.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class1/img4.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class1/img5.jpg

!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class2/img1.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class2/img2.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class2/img3.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class2/img4.jpg
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O /content/dummy_data/val/class2/img5.jpg


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
save_file(json_data, "/content/vissl/configs/config/dataset_catalog.json", append_to_json=False)

from vissl.data.dataset_catalog import VisslDatasetCatalog

# list all the datasets that exist in catalog
print(VisslDatasetCatalog.list())

# get the metadata of dummy_data_folder dataset
print(VisslDatasetCatalog.get("dummy_data_folder"))

!python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config=benchmark/fulltune/imagenet1k/eval_resnet_8gpu_transfer_in1k_fulltune.yaml \
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
    config.CHECKPOINT.DIR="/content/checkpoints" \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE="/content/resnet50-19c8e357.pth" \
    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=""


ls /content/checkpoints/

cat /content/checkpoints/metrics.json
