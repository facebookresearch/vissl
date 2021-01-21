Getting Started with VISSL
==========================

This document provides a brief introduction of usage of built-in command line tools provided by VISSL.


Quick Start with VISSL
---------------------------------

We provide a quick overview for training SimCLR self-supervised model on 1-gpu with VISSL.

Install VISSL
------------------
For installation, please follow our `installation instructions <https://github.com/facebookresearch/vissl/blob/master/INSTALL.md>`_.

Setup dataset
--------------------------
We will use ImageNet-1K dataset and assume the downloaded data to look like:

.. code-block:: bash

    imagenet_full_size
	|_ train
	|  |_ <n0......>
	|  |  |_<im-1-name>.JPEG
	|  |  |_...
	|  |  |_<im-N-name>.JPEG
	|  |_ ...
	|  |_ <n1......>
	|  |  |_<im-1-name>.JPEG
	|  |  |_...
	|  |  |_<im-M-name>.JPEG
	|  |  |_...
	|  |  |_...
	|_ val
	|  |_ <n0......>
	|  |  |_<im-1-name>.JPEG
	|  |  |_...
	|  |  |_<im-N-name>.JPEG
	|  |_ ...
	|  |_ <n1......>
	|  |  |_<im-1-name>.JPEG
	|  |  |_...
	|  |  |_<im-M-name>.JPEG
	|  |  |_...
	|  |  |_...


Running SimCLR Pre-training on 1-gpu
------------------------------------------

We provide a config to train model using the pretext SimCLR task on the ResNet50 model.
Change the :code:`DATA.TRAIN.DATA_PATHS` path to the ImageNet train dataset folder path.

.. code-block:: bash

    python3 run_distributed_engines.py \
        hydra.verbose=true \
    	config.DATA.TRAIN.DATASET_NAMES=[imagenet1k_folder] \
   	config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    	config.DATA.TRAIN.DATA_PATHS=["/path/to/my/imagenet/folder/train"] \
    	config=test/integration_test/quick_simclr \
    	config.CHECKPOINT.DIR="./checkpoints" \
    	config.TENSORBOARD_SETUP.USE_TENSORBOARD=true
