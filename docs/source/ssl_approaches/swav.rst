Train SwAV model
===============================

Author: mathilde@fb.com

VISSL reproduces the self-supervised approach called :code:`SwAV` **Unsupervised learning of visual features by contrasting cluster assignments** which was proposed by
**Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin** in `this paper <https://arxiv.org/abs/2006.09882>`_. SwAV clusters the features while enforcing consistency between
cluster assignments produced for different augmentations (or “views”) of the same image, instead of comparing features directly as in contrastive learning.

How to train SwAV model
----------------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including loss, data augmentations, collators etc required for this approach.

To train ResNet-50 model on 8-gpus on ImageNet-1K dataset using feature projection dimension 128 for memory:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet


Using Synchronized BatchNorm for training
--------------------------------------------

For training SwAV models, we convert all the BatchNorm layers to Global BatchNorm. For this, VISSL supports `PyTorch SyncBatchNorm <https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html>`_
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

SwAV approach leverages mixed precision training by default for better training speed and reducing the model memory requirement.
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

SwAV training uses LARC from `NVIDIA's Apex LARC <https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py>`_. To use LARC, users need to set config option
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
temperature 0.2 for logits, epsilon of 0.04, the training command would look like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        config.LOSS.swav_loss.temperature=0.2 \
        config.LOSS.swav_loss.epsilon=0.04

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    swav_loss:
      temperature: 0.1
      use_double_precision: False
      normalize_last_layer: True
      num_iters: 3
      epsilon: 0.05
      temp_hard_assignment_iters: 0
      crops_for_assign: [0, 1]
      embedding_dim: 128            # automatically inferred from HEAD params
      num_crops: 2                  # automatically inferred from data transforms
      num_prototypes: [3000]        # automatically inferred from model HEAD settings
      # for dumping the debugging info in case loss becomes NaN
      output_dir: ""                # automatically inferred and set to checkpoint dir
      queue:
        start_iter: 0
        queue_length: 0             # automatically adjusted to ensure queue_length % global batch size = 0
        local_queue_length: 0       # automatically inferred to queue_length // world_size

Training different model architecture
----------------------------------------
VISSL supports many backbone architectures including ResNe(X)ts, wider ResNets. Some examples below:


* **Train ResNet-101:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101

* **Train ResNet-50-w2 (2x wider):**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101 \
        config.MODEL.TRUNK.RESNETS.WIDTH_MULTIPLIER=2

* **Train RegNetY-400MF:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        config.MODEL.TRUNK.NAME=regnet config.MODEL.TRUNK.REGNET.name=regnet_y_400mf


* **Train RegNetY-256GF:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        config.MODEL.TRUNK.NAME=regnet \
        config.MODEL.TRUNK.REGNET.depth=27 \
        config.MODEL.TRUNK.REGNET.w_0=640 \
        config.MODEL.TRUNK.REGNET.w_a=230.83 \
        config.MODEL.TRUNK.REGNET.w_m=2.53 \
        config.MODEL.TRUNK.REGNET.group_width=373 \
        config.MODEL.HEAD.PARAMS=[["swav_head", {"dims": [10444, 10444, 128], "use_bn": False, "num_clusters": [3000]}]]


Training with Multi-Crop data augmentation
------------------------------------------------

SwAV is trained using the multi-crop augmentation proposed in `SwAV paper <https://arxiv.org/abs/2006.09882>`_.

Multi-crop augmentation can allow using more positives and also positives of different resolutions. In order to train SwAV with multi-crop
augmentation say crops :code:`2x224 + 4x96` i.e. 2 crops of resolution 224 and 4 crops of resolution 96, the training command looks like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        +config/pretrain/swav/transforms=multicrop_2x224_4x96

The :code:`multicrop_2x224_4x96.yaml` configuration file changes the number of crop settings to 6 crops and the right resolution.

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

By default, the original SwAV approach used the 2-layer MLP-head similar to SimCLR approach. VISSL allows attaching any different desired head. In order to modify the MLP head (more layers, different dimensions etc),
see the following examples:

- **3-layer MLP head:** Use the following head (example for ResNet model)

.. code-block:: yaml

    MODEL:
      HEAD:
        PARAMS: [
          ["swav_head", {"dims": [2048, 2048, 2048, 128], "use_bn": True, "num_clusters": [3000]}],
        ]

- **Use 2-layer MLP with hidden dimension 4096:** Use the following head (example for ResNet model)

.. code-block:: yaml

    MODEL:
      HEAD:
        PARAMS: [
          ["swav_head", {"dims": [2048, 4096, 128], "use_bn": True, "num_clusters": [3000]}],
        ]

Vary the number of epochs
------------------------------------------------

In order to vary the number of epochs to use for training SwAV models, one can achieve this simply
from command line. For example, to train the SwAV model for 100 epochs instead, pass the :code:`num_epochs`
parameter from command line:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        config.OPTIMIZER.num_epochs=100


Vary the number of gpus
----------------------------

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the SwAV model on 4 machines (32gpus)
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1 config.DISTRIBUTED.NUM_NODES=1


* **Training on 4 machines i.e. 32-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=4


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
SwAV using DeepClusterV2 approach and the benchmarks.


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
