Installation
==================

Our installation is simple. We provide pre-built binaries (pip, conda) and also easy instructions for building from source (pip, conda).

Requirements
----------------

At a high level, project requires following system dependencies.

- Linux
- Python>=3.6.2 and <3.9
- PyTorch>=1.4, PyTorch<=3.9.1
- torchvision (matching PyTorch install)
- CUDA (must be a version supported by the pytorch version)
- OpenCV (optional)

Installing VISSL from source (recommended)
-------------------------------------------
The following instructions assume that you have desired CUDA version installed and working.

Install from source in PIP environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Step 1: Create Virtual environment (pip)**

.. code-block:: bash

    python3 -m venv ~/venv
    . ~/venv/bin/activate


- **Step 2: Install PyTorch (pip)**

.. code-block:: bash

    pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html


- **Step 3: Install APEX (pip)**

.. code-block:: bash

    pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py37_cu101_pyt171/download.html apex


- **Step 4: Install VISSL**

.. code-block:: bash

    # clone vissl repository
    cd $HOME && git clone --recursive https://github.com/facebookresearch/vissl.git && cd $HOME/vissl/

    # Optional, checkout stable v0.1.6 branch. While our docs are versioned, the tutorials
    # use v0.1.6 and the docs are more likely to be up-to-date.
    git checkout v0.1.6
    git checkout -b v0.1.6

    # install vissl dependencies
    pip install --progress-bar off -r requirements.txt
    pip install opencv-python
    # update classy vision install to current main branch
    pip uninstall -y classy_vision
    pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/main
    # install vissl dev mode (e stands for editable)
    pip install -e ".[dev]"
    # verify installation
    python -c 'import vissl, apex, cv2'


Install from source in Conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Step 1: Create Conda environment**

If you don't have anaconda, `run this bash scrip to install conda <https://github.com/facebookresearch/vissl/blob/main/docker/common/install_conda.sh>`_.

.. code-block:: bash

    conda create -n vissl_env python=3.7
    source activate vissl_env


- **Step 2: Install PyTorch (conda)**

.. code-block:: bash

    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch


- **Step 3: Install APEX (conda)**

.. code-block:: bash

    conda install -c vissl apex


- **Step 4: Install VISSL**
Follow step4 instructions from the PIP installation above.


Installing VISSL from pre-built binaries
-------------------------------------------

Install VISSL conda package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This assumes you have conda 10.2.

.. code-block:: bash

    conda create -n vissl python=3.8
    conda activate vissl
    conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
    conda install -c vissl -c iopath -c conda-forge -c pytorch -c defaults apex vissl

For all other versions of PyTorch, Python, CUDA, please modify the above instructions with the
desired version. VISSL provides Apex packages for all combinations of pytorch, python and compatible cuda.

Install VISSL pip package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example is with pytorch 1.5.1 and cuda 10.1. Please modify the PyTorch version, cuda version and matching apex version below for the desired settings.

- **Step 1: Create Virtual environment (pip)**

.. code-block:: bash

    python3 -m venv ~/venv
    . ~/venv/bin/activate


- **Step 2: Install PyTorch, OpenCV and APEX (pip)**

  * We use PyTorch=1.5.1 with CUDA 10.1 in the following instruction (user can chose their desired version).
  * There are several ways to install opencv, one possibility is as follows.
  * For APEX, we provide pre-built binary built with optimized C++/CUDA extensions provided by APEX.

.. code-block:: bash

    pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    pip install opencv-python
    pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu101_pyt151/download.html apex


Note that, for the APEX install, you need to get the versions of CUDA, PyTorch, and Python correct in the URL. We provide APEX versions with all possible combinations of Python, PyTorch, CUDA. Select the right APEX Wheels if you desire a different combination.

On Google Colab, everything until this point is already set up. You install APEX there as follows.

.. code-block:: python

    import sys
    import torch
    version_str="".join([
        f"py3{sys.version_info.minor}_cu",
        torch.version.cuda.replace(".",""),
        f"_pyt{torch.__version__[0:5:2]}"
    ])
    !pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/{version_str}/download.html apex


- **Step 3: Install VISSL**

.. code-block:: bash

    pip install vissl
    # verify installation
    python -c 'import vissl'

That's it! You are now ready to use Vissl.

- **Optional: Install Apex from source (common for both pip and conda)**

Apex installation requires that you have a latest nvcc so the c++ extensions can be compiled with latest gcc (>=7.4). Check the APEX website for more instructions.

.. code-block:: bash

    # see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
    # to select cuda architecture you want to build
    CUDA_VER=10.1 TORCH_CUDA_ARCH_LIST="5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.5" ./docker/common/install_apex.sh
