# Installation

Our installation is simple and we provide pre-built binaries (pip, conda) and also instructions for building from source (pip, conda).

## Requirements

At a high level, project requires following dependencies. All these are installable with pip or conda packages.

- Linux
- Python>=3.6
- PyTorch 1.4 or 1.5
- torchvision (matching PyTorch install)
- CUDA at least 10.1
- Hydra 1.0
- Tensoraboard 1.14 (optional)
- Apex (optional)
- Classy Vision
- scikit-learn
- scipy
- opencv

## Installation from source in PIP environment

### Step 1: Create Virtual environment (pip)
```bash
python3 -m venv ~/venv
. ~/venv/bin/activate
```

### Step 2: Install dependencies (pip)
```bash
pip install scipy cython opencv-python scikit-learn
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

## Step 3: Install VISSL

```bash
cd $HOME
git clone --recursive https://github.com/facebookresearch/vissl.git
cd $HOME/vissl/
pip install -e .[dev]  # for dev mode (e stands for editable)
pip install .  # otherwise
# verify installation
python -c 'import vissl'
```

## Installation from source in Conda environment

### Step 1: Create Conda environment

```bash
module load anaconda3/5.0.1
# now, verify your conda installation and check the version
which conda
# now, let's create a conda environment which we will work in
conda create --name vissl python=3.6
# for a faster environment, you can clone the following instead
# conda create --name vissl --file /private/home/prigoyal/mathilde_env.txt
source activate vissl
```

## Step 2: Install Dependencies

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -yq opencv scipy scikit-learn
pip install tensorboard==1.14.0 hydra-core==1.0.0rc1
```

## Step 3: Install VISSL
Follow step3 from the PIP installation.

That's it! You are now ready to use this code.

## Optional Dependency: Install Apex

Apex installation has 2 requirements:
1. NVCC compiler: Pytorch doesn't provide us NVCC compiler but only the cuda toolkit. So install the nvcc which is required to build Apex cuda code.

2. GCC version <=7.3: if using pip, you must have gcc 7.1 to 7.3 for the installation to be successful. Follow https://askubuntu.com/a/915737 for installing gcc7.1

```bash
module load cuda/10.1
# now, check the nvcc is available. the following command should print nvcc path
which nvcc
# to make the code work for both P100 and V100 gpus, export the following env variable
export TORCH_CUDA_ARCH_LIST="6.0;7.0"
# install apex now (note that we recommend a specific apex version for stability)
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex@https://github.com/NVIDIA/apex/tarball/1f2aa9156547377a023932a1512752c392d9bbdf
# Verify apex installed
python -c 'import apex'   # should run and return nothing
```
