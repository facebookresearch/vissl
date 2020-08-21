# Installation

Our installation is simple and we provide pre-built binaries (pip, conda) and also instructions for building from source (pip, conda).

## Requirements

At a high level, project requires following system dependencies.

- Linux
- Python>=3.6
- PyTorch 1.4 or 1.5
- torchvision (matching PyTorch install)
- CUDA at least 9.2 (optional)

## Installation from source in PIP environment

### Step 1: Create Virtual environment (pip)
```bash
python3 -m venv ~/venv
. ~/venv/bin/activate
```

### Step 2: Install PyTorch (pip)
```bash
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Step 3: Install VISSL

```bash
cd $HOME && git clone --recursive https://github.com/facebookresearch/vissl.git && cd $HOME/vissl/
pip install -e .[dev]  # for dev mode (e stands for editable)
pip install .  # otherwise
# verify installation
python -c 'import vissl'
```

## Installation from source in Conda environment

### Step 1: Create Conda environment

```bash
# install conda
./docker/common/install_conda.sh
source activate vissl_env
```

## Step 2: Install PyTorch (conda)

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

## Step 3: Install VISSL
Follow step3 from the PIP installation.

That's it! You are now ready to use this code.

## Optional Dependency: Install Apex (common for both pip and conda)

Apex installation requires that you have a latest nvcc so the c++ extensions can be compiled with latest gcc (>=7.4). Check the APEX website for more instructions.

```bash
# see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list to select cuda architecture you want to build
CUDA_VER=10.1 TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.5" ./docker/common/install_apex.sh
```
