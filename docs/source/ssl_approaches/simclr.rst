Train SimCLR model
===============================

VISSL reproduces the self-supervised approach **A Simple Framework for Contrastive Learning of Visual Representations** proposed by **Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton** in `this paper <https://arxiv.org/abs/2002.05709>`_.

How to train SimCLR model
---------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including loss, data augmentations, collators etc required for this approach.

To train ResNet-50 model on 4-machines (8-nodes) on ImageNet-1K dataset with SimCLR approach using MLP-head, loss temperature of 0.1 and feature projection dimension 128:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/simclr/simclr_8node_resnet


Using Synchronized BatchNorm for training
--------------------------------------------

For training SimCLR models, we convert all the BatchNorm layers to Global BatchNorm. For this, VISSL supports `PyTorch SyncBatchNorm <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html>`_
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

Using LARC for training
--------------------------------------------

SimCLR training uses LARC from `NVIDIA's Apex LARC <https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py>`_. To use LARC, users need to set config option
:code:`OPTIMIZER.use_larc=True`. VISSL exposed LARC parameters that users can tune. Full list of LARC parameters exposed by VISSL:

.. code-block:: yaml

    OPTIMIZER:
      name: "sgd"
      use_larc: False  # supported for SGD only for now
      larc_config:
        clip: False
        eps: 1e-08
        trust_coefficient: 0.001

.. note::

    LARC is currently supported for SGD optimizer only.

Vary the training loss settings
------------------------------------------------
Users can adjust several settings from command line to train the model with different hyperparams. For example: to use a different
temperature 0.2 for logits and different output projection dimension of 256:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/simclr/simclr_8node_resnet \
        config.LOSS.simclr_info_nce_loss.temperature=0.2 \
        config.LOSS.simclr_info_nce_loss.buffer_params.embedding_dim=256

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    simclr_info_nce_loss:
      temperature: 0.1
      buffer_params:
        embedding_dim: 128
        world_size: 64                # automatically inferred
        effective_batch_size: 4096    # automatically inferred


Training different model architecture
------------------------------------------------
VISSL supports many backbone architectures including ResNe(X)ts, wider ResNets. Some examples below:


* **Train ResNet-101:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/simclr/simclr_8node_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101

* **Train ResNet-50-w2 (2x wider):**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/simclr/simclr_8node_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101 \
        config.MODEL.TRUNK.RESNETS.WIDTH_MULTIPLIER=2


Training with Multi-Crop data augmentation
------------------------------------------------

The original SimCLR approach is proposed for 2 positives per image. We expand the SimCLR approach to work for more positives following the multi-crop
augmentation proposed in SwAV paper. See SwAV paper https://arxiv.org/abs/2006.09882 for the multi-crop augmentation details.

Multi-crop augmentation can allow using more positives and also positives of different resolutions for SimCLR. VISSL provides
a version of SimCLR loss for multi-crop training :code:`multicrop_simclr_info_nce_loss`. In order to train SimCLR with multi-crop
augmentation say crops :code:`2x160 + 4x96` i.e. 2 crops of resolution 160 and 4 crops of resolution 96, the training command looks like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/simclr/simclr_8node_resnet \
        +config/pretrain/simclr/transforms=multicrop_2x160_4x96

The :code:`multicrop_2x160_4x96.yaml` configuration file changes 2 things:

- Transforms: Simply replace the :code:`ImgReplicatePil` transform (which creates 2 copies of image) with :code:`ImgPilToMultiCrop` which creates multi-crops of multiple resolutions.

- Loss: Use the loss :code:`multicrop_simclr_info_nce_loss` instead which inherits from :code:`simclr_info_nce_loss` and modifies the loss to work for multi-crop input.

Varying the multi-crop augmentation settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL allows modifying the crops to use. Full settings exposed:

.. code-block:: yaml

    TRANSFORMS:
      - name: ImgPilToMultiCrop
        total_num_crops: 6                      # Total number of crops to extract
        num_crops: [2, 4]                       # Specifies the number of type of crops.
        size_crops: [160, 96]                   # Specifies the height (height = width) of each patch
        crop_scales: [[0.08, 1], [0.05, 0.14]]  # Scale of the crop


Varying the multi-crop loss settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    multicrop_simclr_info_nce_loss:
      temperature: 0.1
      num_crops: 2                      # automatically inferred from data transforms
      buffer_params:
        world_size: 64                  # automatically inferred
        embedding_dim: 128
        effective_batch_size: 4096      # automatically inferred


Training with different MLP head
------------------------------------------------

Original SimCLR approach used 2-layer MLP head. VISSL allows attaching any different desired head. In order to modify the MLP head (more layers, different dimensions etc),
see the following examples:

- **3-layer MLP head:** Use the following head (example for ResNet model)

.. code-block:: yaml

    MODEL:
      HEAD:
        PARAMS: [
          ["mlp", {"dims": [2048, 2048], "use_relu": True}],
          ["mlp", {"dims": [2048, 2048], "use_relu": True}],
          ["mlp", {"dims": [2048, 128]}],
        ]

- **Use 2-layer MLP with hidden dimension 4096:** Use the following head (example for ResNet model)

.. code-block:: yaml

    MODEL:
      HEAD:
        PARAMS: [
          ["mlp", {"dims": [2048, 4096], "use_relu": True}],
          ["mlp", {"dims": [4096, 128]}],
        ]


Vary the number of epochs
------------------------------------------------

In order to vary the number of epochs to use for training SimCLR models, one can achieve this simply
from command line. For example, to train the SimCLR model for 100 epochs instead, pass the :code:`num_epochs`
parameter from command line:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/simclr/simclr_8node_resnet \
        config.OPTIMIZER.num_epochs=100


Vary the number of gpus
------------------------------------------------

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the SimCLR model on 8-gpus
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/simclr/simclr_8node_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1 config.DISTRIBUTED.NUM_NODES=1


* **Training on 8-gpus:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/simclr/simclr_8node_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=1


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
VISSL for SimCLR and the benchmarks.


Citations
---------

.. code-block:: none

    @misc{chen2020simple,
        title={A Simple Framework for Contrastive Learning of Visual Representations},
        author={Ting Chen and Simon Kornblith and Mohammad Norouzi and Geoffrey Hinton},
        year={2020},
        eprint={2002.05709},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
