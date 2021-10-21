Train Barlow Twins model
========================

VISSL reproduces the self-supervised approach **Barlow Twins: Self-Supervised Learning
via Redundancy Reduction** proposed by **Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun,
Stéphane Deny** in `this paper <https://arxiv.org/abs/2103.03230v1>`_.

How to train Barlow Twins model
-------------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to
reproduce the model. VISSL implements all the components including loss, data
augmentations, collators etc required for this approach.

To train ResNet-50 model on 4 octo-gpu nodes on ImageNet-1K dataset with Barlow Twins approach using a 8192-8192-8192 MLP-head, lambda of 0.0051 and scale loss of 0.024:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/barlow_twins/barlow_twins_4node_resnet


Using Synchronized BatchNorm for training
--------------------------------------------

For training Barlow Twins models, we convert all the BatchNorm layers to Global BatchNorm. For this, VISSL supports `PyTorch SyncBatchNorm <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html>`_
module and `NVIDIA's Apex SyncBatchNorm <https://nvidia.github.io/apex/_modules/apex/parallel/optimized_sync_batchnorm.html>`_ layers. Set the config params :code:`MODEL.SYNC_BN_CONFIG.SYNC_BN_TYPE` to :code:`apex` or :code:`pytorch`.

If you want to use Apex, VISSL provides :code:`anaconda` and :code:`pip` packages of Apex (compiled with Optimzed C++ extensions/CUDA kernels). The Apex
packages are provided for all versions of :code:`CUDA (9.2, 10.0, 10.1, 10.2, 11.0), PyTorch >= 1.4 and Python >=3.6 and <=3.9`.

To use SyncBN during training, one needs to set the following parameters in configuration file:

.. code-block:: yaml

    MODEL:
      SYNC_BN_CONFIG:
        CONVERT_BN_TO_SYNC_BN: True
        SYNC_BN_TYPE: apex
        # 1) if group_size=-1 -> use the VISSL default setting. We synchronize within a
        #     machine and hence will set group_size=num_gpus per node. This gives the best
        #     speedup.
        # 2) if group_size>0 -> will set group_size=value set by user.
        # 3) if group_size=0 -> no groups are created and process_group=None. This means
        #     global sync is done.
        GROUP_SIZE: 8


Vary the training loss settings
------------------------------------------------
Users can adjust several settings from command line to train the model with different hyperparams. For example: to use a different
lambda of 0.01:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/barlow_twins/barlow_twins_4node_resnet \
        config.LOSS.barlow_twins_loss.lambda_=0.01

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    barlow_twins_loss:
      lambda_: 0.01
      scale_loss: 0.024
      embedding_dim: 8192


Training different model architecture
------------------------------------------------
VISSL supports many backbone architectures including ResNe(X)ts, wider ResNets. Some examples below:


* **Train ResNet-101:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/barlow_twins/barlow_twins_4node_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101

* **Train ResNet-50-w2 (2x wider):**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/barlow_twins/barlow_twins_4node_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101 \
        config.MODEL.TRUNK.RESNETS.WIDTH_MULTIPLIER=2


Vary the number of gpus
------------------------------------------------

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the SimCLR model on 8-gpus
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/barlow_twins/barlow_twins_4node_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1 config.DISTRIBUTED.NUM_NODES=1


* **Training on 8-gpus:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/barlow_twins/barlow_twins_4node_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=1


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
VISSL for Barlow Twins and the benchmarks.


Citations
---------

.. code-block:: none

    @misc{zbontar2021barlow,
      title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
      author={Jure Zbontar and Li Jing and Ishan Misra and Yann LeCun and Stéphane Deny},
      year={2021},
      eprint={2103.03230},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
