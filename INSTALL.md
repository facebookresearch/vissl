# Installation

Our installation is simple and anaconda3 based. Follow the steps below:

**Requirements**: NVIDIA GPU (P100 and above), Linux

**Note:** We currently do not provide support for CPU only runs except SVM trainings.


## Step 1: Clone the github repo

```bash
cd $HOME
git clone --recursive https://github.com/fairinternal/ssl_scaling.git
cd $HOME/ssl_scaling
# checkout any branches here if you want to
git submodule update --init
```

## Step 2: Install Anaconda3

```bash
module load anaconda3/5.0.1
```

Now, verify your conda installation and check the version:

```bash
which conda
```

This command should print the path of your conda bin. If it doesn't, make sure conda is in your $PATH.

Now, let's create a conda environment which we will work in.

```bash
conda create --name ssl_framework python=3.6
source activate ssl_framework
```

## Step 3: Install Pytorch

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

## Step 4: Install Apex

Pytorch doesn't provide us NVCC compiler but only the cuda toolkit. So install the
nvcc which is required to build Apex cuda code.

```bash
module load cuda/10.1
```

Now, check the nvcc is available.
```bash
which nvcc
```

This command should print nvcc bin path. Now, we are ready to install Apex.
(Instructions from https://github.com/NVIDIA/apex#linux)

```bash
cd $HOME/ssl_scaling/third-party/apex

# to make the code work for both P100 and V100 gpus, export the following env variable
export TORCH_CUDA_ARCH_LIST="6.0;7.0"

pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Verify apex installed:
```bash
cd $HOME/ssl_scaling
python -c 'import apex'   # should run and return nothing
```

## Step 5: Install other dependencies

```bash
conda install -yq opencv scipy  scikit-learn
```


## Step 6: Install Classy Vision

```bash
cd $HOME/ssl_scaling/third-party/ClassyVision/
pip install .
```

Verify classy_vision installed correctly:
```bash
cd $HOME/ssl_scaling
python -c 'import classy_vision'  # should run and return nothing
```

## Step 7: Install SSL framework

```bash
cd $HOME/ssl_scaling/
# NOTE: if you want to build in dev mode, replace "install" with "develop"
python setup.py install
```

### Step 8: Verify installation

```bash
python -c 'import vissl'
```

That's it! You are now ready to use this code.
