Train RotNet model
===============================

VISSL reproduces the self-supervised approach **Unsupervised Representation Learning by Predicting Image Rotations**
proposed by **Spyros Gidaris, Praveer Singh, Nikos Komodakis** in `this paper <https://arxiv.org/abs/1803.07728>`_.

How to train RotNet model
---------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including data augmentations, collators, etc required for this approach.

To train ResNet-50 model on 8-gpus on ImageNet-1K dataset using 4 rotation angles:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/rotnet/rotnet_8gpu_resnet


Training different model architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VISSL supports many backbone architectures including AlexNet, ResNe(X)ts. Some examples below:

* **Train AlexNet model**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/rotnet/rotnet_8gpu_resnet \
        config.MODEL.TRUNK.NAME=alexnet_rotnet


* **Train ResNet-101:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/rotnet/rotnet_8gpu_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101



Vary the number of gpus
~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the RotNet model on 4 machines (32gpus)
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/rotnet/rotnet_8gpu_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1


* **Training on 4 machines i.e. 32-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/rotnet/rotnet_8gpu_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=4


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
VISSL using RotNet approach and the benchmarks.


Citation
---------

.. code-block:: none

    @misc{gidaris2018unsupervised,
        title={Unsupervised Representation Learning by Predicting Image Rotations},
        author={Spyros Gidaris and Praveer Singh and Nikos Komodakis},
        year={2018},
        eprint={1803.07728},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
