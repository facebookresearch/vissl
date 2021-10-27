
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

!wget -q -O /content/resnet_simclr.torch https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch

from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict

# Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
# All other options override the simclr_8node_resnet.yaml config.

cfg = [
  'config=pretrain/simclr/simclr_8node_resnet.yaml',
  'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/content/resnet_simclr.torch', # Specify path for the model weights.
  'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
  'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
  'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
  'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
  'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]' # Extract only the res5avg features.
]

# Compose the hydra configuration.
cfg = compose_hydra_configuration(cfg)
# Convert to AttrDict. This method will also infer certain config options
# and validate the config is valid.
_, cfg = convert_to_attrdict(cfg)

from vissl.models import build_model

model = build_model(cfg.MODEL, cfg.OPTIMIZER)

from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights

# Load the checkpoint weights.
weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)


# Initializei the model with the simclr model weights.
init_model_from_consolidated_weights(
    config=cfg,
    model=model,
    state_dict=weights,
    state_dict_key_name="classy_state_dict",
    skip_layers=[],  # Use this if you do not want to load all layers
)

print("Weights have loaded")

!wget -q -O /content/test_image.jpg https://raw.githubusercontent.com/facebookresearch/vissl/master/.github/logo/Logo_Color_Light_BG.png

from PIL import Image
import torchvision.transforms as transforms

def extract_features(path):
  image = Image.open(path)

  # Convert images to RGB. This is important
  # as the model was trained on RGB images.
  image = image.convert("RGB")

  # Image transformation pipeline.
  pipeline = transforms.Compose([
      transforms.CenterCrop(224),
      transforms.ToTensor(),
  ])
  x = pipeline(image)

  features = model(x.unsqueeze(0))

  features_shape = features[0].shape

  print(f"Features extracted have the shape: { features_shape }")

extract_features("/content/test_image.jpg")

!wget -q -O /content/resnet_in1k.torch https://dl.fbaipublicfiles.com/vissl/model_zoo/sup_rn50_in1k_ep105_supervised_8gpu_resnet_17_07_20.733dbdee/model_final_checkpoint_phase208.torch

from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict

from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict

# Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
# All other options override the simclr_8node_resnet.yaml config.

# Note here we freeze the trunk and the head, and specify that we want to eval
# with the trunk and head.
cfg = [
  'config=pretrain/supervised/supervised_1gpu_resnet_example.yaml',
  'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/content/resnet_in1k.torch', # Specify path for the model weights.
  'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
  'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_AND_HEAD=True', # Freeze trunk. 
  'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD=True', # Extract the trunk features, as opposed to the HEAD.
]

# NOTE: After this everything is the same as the above example of extracting 
# the TRUNK features.

# Compose the hydra configuration.
cfg = compose_hydra_configuration(cfg)

# Convert to AttrDict. This method will also infer certain config options
# and validate the config is valid.
_, cfg = convert_to_attrdict(cfg)

# Build the model
from vissl.models import build_model
from vissl.utils.checkpoint import init_model_from_consolidated_weights

model = build_model(cfg.MODEL, cfg.OPTIMIZER)

# Load the checkpoint weights.
weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

# Initializei the model with the simclr model weights.
init_model_from_consolidated_weights(
    config=cfg,
    model=model,
    state_dict=weights,
    state_dict_key_name="classy_state_dict",
    skip_layers=[],  # Use this if you do not want to load all layers
)

print("Weights have loaded")

extract_features("/content/test_image.jpg")
