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
pip install scipy cython opencv-python scikit-learn tensorboard==1.14.0 hydra-core==1.0.0rc1
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 3: Clone VISSL from github
```bash
cd $HOME
git clone --recursive https://github.com/facebookresearch/vissl.git
cd $HOME/vissl
# checkout submodules as well (necessary if 1st time checkout)
git submodule update --init --recursive
```

### Step 4: Install Apex

Pytorch doesn't provide us NVCC compiler but only the cuda toolkit. So install the
nvcc which is required to build Apex cuda code.

NOTE: if using pip, uou must have gcc 7.1 to 7.3 for the installation to be successful. Follow https://askubuntu.com/a/915737 for installing gcc7.1

```bash
module load cuda/10.1
# now, check the nvcc is available
which nvcc
```

This command should print nvcc bin path. Now, we are ready to install Apex.
(Instructions from https://github.com/NVIDIA/apex#linux)

```bash
cd $HOME/vissl/third-party/apex
# to make the code work for both P100 and V100 gpus, export the following env variable
export TORCH_CUDA_ARCH_LIST="6.0;7.0"
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# Verify apex installed
cd $HOME/vissl
python -c 'import apex'   # should run and return nothing
```

### Step 5: Install ClassyVision

```bash
cd $HOME/vissl/third-party/ClassyVision/
pip install .
```

Verify classy_vision installed correctly:
```bash
cd $HOME/vissl
python -c 'import classy_vision'  # should run and return nothing
```

## Step 6: Install VISSL

```bash
cd $HOME/vissl/
pip install -e .[dev]  # for dev mode (e stands for editable)
pip setup.py .  # otherwise
# verify installation
python -c 'import vissl'
```
# ------------------------------------------------------------------------------------------- #
## Installation from source in Conda environment

### Step 1: Install Anaconda3

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

## Step 3: Install Apex and Classy Vision and VISSL
Follow steps 3 to 6 from the PIP installation.


That's it! You are now ready to use this code.

