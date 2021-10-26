Train DeepCluster V2 model
===============================

Author: mathilde@fb.com

VISSL reproduces the self-supervised approach called :code:`DeepClusterV2` which is an improved version of original `DeepCluster approach <https://arxiv.org/abs/1807.05520>`_. The DeepClusterV2 approach was proposed in work **Unsupervised learning of visual features by contrasting cluster assignments** by
**Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin** in `this paper <https://arxiv.org/abs/2006.09882>`_. DeepClusterV2
combines the benefits of `DeepCluster <https://arxiv.org/abs/1807.05520>`_ and `NPID <https://arxiv.org/abs/1805.01978>`_ approaches.

How to train DeepClusterV2 model
----------------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including loss, data augmentations, collators etc required for this approach.

To train ResNet-50 model on 8-gpus on ImageNet-1K dataset using feature projection dimension 128 for memory:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet


Using Synchronized BatchNorm for training
--------------------------------------------

For training DeepClusterV2 models, we convert all the BatchNorm layers to Global BatchNorm. For this, VISSL supports `PyTorch SyncBatchNorm <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html>`_
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

Using Mixed Precision for training
--------------------------------------------

DeepClusterV2 approach leverages mixed precision training by default for better training speed and reducing the model memory requirement.
For this, we use `NVIDIA Apex Library with Apex AMP level O1 <https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use>`_.

To use Mixed precision training, one needs to set the following parameters in configuration file:

.. code-block:: yaml

    MODEL:
      AMP_PARAMS:
        USE_AMP: True
        # Use O1 as it is robust and stable than O3. If you want to use O3, we recommend
        # the following setting:
        # {"opt_level": "O3", "keep_batchnorm_fp32": True, "master_weights": True, "loss_scale": "dynamic"}
        AMP_ARGS: {"opt_level": "O1"}

Using LARC for training
--------------------------------------------

DeepClusterV2 training uses LARC from `NVIDIA's Apex LARC <https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py>`_. To use LARC, users need to set config option
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
---------------------------------
Users can adjust several settings from command line to train the model with different hyperparams. For example: to use a different
temperature 0.2 for logits, projection dimension 256 for the embedding, the training command would look like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet \
        config.LOSS.deepclusterv2_loss.temperature=0.2 \
        config.LOSS.deepclusterv2_loss.memory_params.embedding_dim=256

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    deepclusterv2_loss:
      DROP_LAST: True             # automatically inferred from DATA.TRAIN.DROP_LAST
      BATCHSIZE_PER_REPLICA: 256  # automatically inferred from DATA.TRAIN.BATCHSIZE_PER_REPLICA
      num_crops: 2                # automatically inferred from DATA.TRAIN.TRANSFORMS
      temperature: 0.1
      num_clusters: [3000, 3000, 3000]
      kmeans_iters: 10
      memory_params:
        crops_for_mb: [0]
        embedding_dim: 128
      # following parameters are auto-filled before the loss is created.
      num_train_samples: -1       # automatically inferred

Training different model architecture
----------------------------------------
VISSL supports many backbone architectures including ResNe(X)ts, wider ResNets. Some examples below:


* **Train ResNet-101:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101

* **Train ResNet-50-w2 (2x wider):**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101 \
        config.MODEL.TRUNK.RESNETS.WIDTH_MULTIPLIER=2

Training with Multi-Crop data augmentation
------------------------------------------------

DeepClusterV2 can be trained for for more positives following the multi-crop
augmentation proposed in SwAV paper. See SwAV paper https://arxiv.org/abs/2006.09882 for the multi-crop augmentation details.

Multi-crop augmentation can allow using more positives and also positives of different resolutions. In order to train DeepClusterV2 with multi-crop
augmentation say crops :code:`2x160 + 4x96` i.e. 2 crops of resolution 160 and 4 crops of resolution 96, the training command looks like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet \
        +config/pretrain/deepcluster_v2/transforms=multicrop_2x160_4x96

The :code:`multicrop_2x160_4x96.yaml` configuration file changes the number of crop settings to 6 crops.

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

Training with different MLP head
------------------------------------------------

By default, the original DeepClusterV2 approach used the 2-layer MLP-head similar to SimCLR approach. VISSL allows attaching any different desired head. In order to modify the MLP head (more layers, different dimensions etc),
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

In order to vary the number of epochs to use for training DeepClusterV2 models, one can achieve this simply
from command line. For example, to train the DeepClusterV2 model for 100 epochs instead, pass the :code:`num_epochs`
parameter from command line:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet \
        config.OPTIMIZER.num_epochs=100


Vary the number of gpus
----------------------------

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the DeepClusterV2 model on 4 machines (32gpus)
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1 config.DISTRIBUTED.NUM_NODES=1


* **Training on 4 machines i.e. 32-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/deepcluster_v2/deepclusterv2_2crops_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=4


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
VISSL using DeepClusterV2 approach and the benchmarks.


Citations
---------

* **DeepClusterV2**

.. code-block:: none

    @misc{caron2020unsupervised,
        title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
        author={Mathilde Caron and Ishan Misra and Julien Mairal and Priya Goyal and Piotr Bojanowski and Armand Joulin},
        year={2020},
        eprint={2006.09882},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
