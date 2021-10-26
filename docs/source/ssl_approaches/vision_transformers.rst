Training Vision Transformer models
============================================

Author: matthew.l.leavitt@gmail.com

VISSL contains implementations of multiple vision transformer model variants:

- **Vision Transformers (ViT)**: Published in |vision_transformer_cite|_, ViT was a breakthrough for transformers in vision tasks, and comprises a set of model architectures and hyperparameters that closely follow `Vaswani et al.'s original implementation of transformers for NLP <https://arxiv.org/abs/1706.03762>`_.

- **Data-efficient image Transformers (DeiT)**: Published in |deit_cite|_, DeiT is architecturally similar to ViT, but is distinguished by its hyperparameters and use of distillation during training. Training with distillation is not currently supported in VISSL, but DeiT provides benefits over ViT even when training without distillation.

- **Convolutional Vision Transformer (ConViT)**: Published in |convit_cite|_, the ConViT was designed with the goal of combining the expressivity of transformers with the sample-efficiency of the convolutional inductive bias. ConViT achieves this by replacing the standard self-attention layers of the vision transformer with "gated positional self-attention" layers that are initialized to perform both convolution and self-attention, and learn the optimal trade-off between the two functions.

.. |vision_transformer_cite| replace:: Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020)
.. _vision_transformer_cite: https://arxiv.org/abs/2010.11929

.. |deit_cite| replace:: Touvron et al., *Training data-efficient image transformers & distillation through attention* (2020)
.. _deit_cite: https://arxiv.org/abs/2012.12877

.. |convit_cite| replace:: d'Ascoli et al., *ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases* (2021)
.. _convit_cite: https://arxiv.org/abs/2103.10697

We start by demonstrating how to train ViTs, and continue with examples of training DeiTs and ConViTs. The hyperparameter provided in this walkthrough and the provided config files reflect the authors' recommendations from the associated publications, when available. However, many of the model-method combinations for which we have config files (e.g. ConViT with SimCLR) have not been examined in published work, so the hyperparameters are what we found to work in preliminary analyses. If you find something that works, please contribute new config files to VISSL!

Supervised ViT training on 1 gpu
--------------------------------------------

VISSL provides yaml configuration files for many of the common hyperparameter settings for different vision transformer model variants. We will start with supervised training of a ViT-baseline (ViT-B) model on the ImageNet-1k dataset (see :ref:`Using Data<Using Data>` to set up your dataset) using a single GPU:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/vision_transformer/supervised/1gpu_vit_example

.. =====  ======  =========== ======== ====================
.. Model  Layers  Trunk Width MLP Size Self-Attention Heads
.. =====  ======  =========== ======== ====================
.. ViT-B    12    768         3072     12
.. ViT-L    24    1024        4096     16
.. ViT-H    32    1280        5120     16
.. =====  ======  =========== ======== ====================

Let's take a look at the ``model`` section of the config file:

.. code-block:: yaml

  MODEL:
    GRAD_CLIP:
      USE_GRAD_CLIP: True
    TRUNK:
      NAME: vision_transformer
      VISION_TRANSFORMERS:
        IMAGE_SIZE: 224
        PATCH_SIZE: 16
        NUM_LAYERS: 12
        NUM_HEADS: 12
        HIDDEN_DIM: 768
        MLP_DIM: 3072
        # MLP and projection layer dropout rate
        DROPOUT_RATE: 0
        # Attention dropout rate
        ATTENTION_DROPOUT_RATE: 0
        # Use the token for classification. Currently no alternatives
        # supported
        CLASSIFIER: token
        # Stochastic depth dropout rate. Turning on stochastic depth and
        # using aggressive augmentation is essentially the difference
        # between a DeiT and a ViT.
        DROP_PATH_RATE: 0
        QKV_BIAS: False # Bias for QKV in attention layers.
        QK_SCALE: False # Scale
    HEAD:
      PARAMS: [
      ["vision_transformer_head", {"in_plane": 768, "hidden_dim": 3072,
                                   "num_classes": 1000}],
      ]

Starting at the top, we can see that ``MODEL.GRAD_CLIP.USE_GRAD_CLIP`` is set to ``True``, indicating that gradients beyond a certain magnitude will be clipped during training, as per the ViT authors' training recipe. What is this "certain magnitude"? You can find all the default config values in ``vissl/config/defaults.yaml``. **Pro-tip**: ``defaults.yaml``'s heavy annotations make it a great resource for figuring out how VISSL works.

Moving on to ``MODEL.TRUNK.NAME``, we can see that we are using a ``vision_transformer``, which corresponds to class of model in ``vissl/models/trunks``. The architectural hyperparameters are contained in ``MODEL.TRUNK.VISION_TRANSFORMERS`` (again, see ``defaults.yaml`` for details on these hyperparameters). These are the appropriate architectural hyperparameters for a ViT-B as per the `publication <https://arxiv.org/abs/2010.11929>`_.

The head architecture is specified in ``MODEL.HEAD.PARAMS``. See the :ref:`models documentation<Building Models>` for more information about how to specify model head parameters. ``"in_plane"`` is the dimensionality of the input to the head, which must match the output dimensionality of the trunk, which for the ViT-B is 768.

Let's move on to the ``OPTIMIZER`` section of the configuration file:

.. code-block:: yaml

  OPTIMIZER:
    name: adamw
    weight_decay: 0.05
    num_epochs: 300
    betas: [.9, .999] # for Adam/AdamW
    param_schedulers:
      lr:
        auto_lr_scaling:
          auto_scale: True
          base_value: 0.0005
          base_lr_batch_size: 1024
        name: composite
        schedulers:
          - name: linear
            start_value: 0.0
            end_value: 0.0005
          - name: cosine
            start_value: 0.0005
            end_value: 0
        interval_scaling: [rescaled, rescaled]
        update_interval: step
        lengths: [0.017, 0.983]
      # Parameters to omit from regularization.
      # We don't want to regularize the class token or position in the ViT.
      non_regularized_parameters: [pos_embedding, class_token]

Again, these hyperparameters reflect the authors' recipe in the original ViT publication. It's also worth pointing out that VISSL offers a lot control of the optimizer, so be sure to :ref:`read up on it<Using Optimizers>` and poke around in ``vissl/config/defaults.yaml``. `AdamW <https://arxiv.org/abs/1711.05101>`_ thus far seems like the most consistently successful optimizer for training vision transformers, so we use it in all our config files.

This config file is for a ViT-B16. What if we wanted instead to train the next larger ViT, ViT-L? This would require the following changes to the model architecture parameters:

.. code-block:: yaml

  MODEL:
    GRAD_CLIP:
      USE_GRAD_CLIP: True
    TRUNK:
      NAME: vision_transformer
      VISION_TRANSFORMERS:
        IMAGE_SIZE: 224
        PATCH_SIZE: 16
        NUM_LAYERS: 24 # Increased from 12->24
        NUM_HEADS: 16 # Increased from 12->16
        HIDDEN_DIM: 1024 # Increased from 768->1024
        MLP_DIM: 4096 # Increased from 3072->4096
        DROPOUT_RATE: 0.1
        ATTENTION_DROPOUT_RATE: 0
        CLASSIFIER: token
        DROP_PATH_RATE: 0
        QKV_BIAS: False # Bias for QKV in attention layers.
        QK_SCALE: False # Scale
    HEAD:
      PARAMS: [
      ["vision_transformer_head", {"in_plane": 1024, "hidden_dim": 4096,
                                   "num_classes": 1000}],
      ] # in_plane increased from -> 768->1024

Changing only these parameters would likely lead to an out-of-memory error due to the size difference between the ViT-B and ViT-L, so we also need to decrease the batch size:

.. code-block:: yaml

  DATA:
    TRAIN:
      BATCHSIZE_PER_REPLICA: 16 # Reduced from 128->32
    ...
    (unchanged parameters skipped for brevity)
    ...
    TEST:
      BATCHSIZE_PER_REPLICA: 64 # Reduced from 256->64


MoCo ViT-B16 training
---------------------
``config/pretrain/vision_transformer/moco/vit_b16.yaml`` is the configuration file for training a ViT-B16 with MoCo. There are a few key differences between this configuration file and the configuration for 1-gpu supervised training. First, the data parameters:

.. code-block:: yaml

  DATA:
    NUM_DATALOADER_WORKERS: 5
    TRAIN:
      DATA_SOURCES: [disk_folder]
      DATASET_NAMES: [imagenet1k_folder]
      BATCHSIZE_PER_REPLICA: 128
      LABEL_TYPE: sample_index    # just an implementation detail. Label isn't used
      TRANSFORMS:
        - name: ImgReplicatePil
          num_times: 2
        - name: RandomResizedCrop
          size: 224
        - name: RandomHorizontalFlip
          p: 0.5
        - name: ImgPilColorDistortion
          strength: 1.0
        - name: ImgPilGaussianBlur
          p: 0.5
          radius_min: 0.1
          radius_max: 2.0
        - name: ToTensor
        - name: Normalize
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
      COLLATE_FUNCTION: moco_collator
      MMAP_MODE: True
      COPY_TO_LOCAL_DISK: False
      COPY_DESTINATION_DIR: /tmp/imagenet1k/
      DROP_LAST: True


Most of the contrastive training schemes require duplicating each sample, which is achieved in this case by using the transformation ``ImgReplicatePil``, which is specified in ``DATA.TRAIN.TRANSFORMS``. Many of the self-supervised methods also require a specific data collator, specified in ``DATA.TRAIN.COLLATE_FUNCTION``. See :ref:`Using Data<Using Data>` for more details.

The `LOSS` section of the config file specifies the parameters for the MoCo loss:

.. code-block:: yaml

  LOSS:
    name: moco_loss
    moco_loss:
      embedding_dim: 128
      queue_size: 65536
      momentum: 0.999
      temperature: 0.2

The output dimensionality of the model head must match ``LOSS.moco_loss.embedding_dim``.

If you move to the bottom of the file, you can see that this file specifies using 32 gpus across 4 machines:

.. code-block:: yaml

  DISTRIBUTED:
    BACKEND: nccl
    NUM_NODES: 4
    NUM_PROC_PER_NODE: 8
    RUN_ID: "60215"
  MACHINE:
    DEVICE: gpu

See the :ref:`documentation on running large jobs<Train on multiple-gpus>` for more details on scaling up!


Training DeiT with SwAV
--------------------------------
This section primarily addresses the differences between DeiT and ViT. See :ref:`here<Train SwAV model>` for detailed information about how to use SwAV. Aside from training with distillation, which is not currently supported in VISSL, the differences between DeiT and ViT are mostly in the choice of hyperparameters (see Table 9 in the `DeiT paper <https://arxiv.org/abs/2012.12877>`_ for details):

.. code-block:: yaml

  MODEL:
    TRUNK:
      NAME: vision_transformer
      VISION_TRANSFORMERS:
        IMAGE_SIZE: 224
        PATCH_SIZE: 16
        NUM_LAYERS: 12
        NUM_HEADS: 16
        HIDDEN_DIM: 768
        MLP_DIM: 3072
        CLASSIFIER: token
        DROPOUT_RATE: 0 # 0.1 for ViT
        ATTENTION_DROPOUT_RATE: 0
        DROP_PATH_RATE: 0.1 # stochastic depth dropout probability. 0 for ViT
        DROP_PATH_RATE: 0
        QKV_BIAS: False # Bias for QKV in attention layers.
        QK_SCALE: False # Scale

The DeiT uses `stochastic depth <https://arxiv.org/abs/1603.09382>`_, which is set via ``MODEL.TRUNK.VISION_TRANSORMERS.DROP_PATH_RATE``. In contrast to ViT, DeiT does not use gradient clipping. DeiT also uses a number of data augmentations:

.. code-block:: yaml

  DATA:
    NUM_DATALOADER_WORKERS: 8
    TRAIN:
      DATA_SOURCES: [disk_folder]
      DATASET_NAMES: [imagenet1k_folder]
      LABEL_TYPE: "zero"
      BATCHSIZE_PER_REPLICA: 16
      DROP_LAST: True
      TRANSFORMS:
        - name: ImgPilToMultiCrop
          total_num_crops: 2
          size_crops: [224]
          num_crops: [2]
          crop_scales: [[0.14, 1]]
        - name: RandomHorizontalFlip
        - name: RandAugment
          magnitude: 9
          magnitude_std: 0.5
          increasing_severity: True
        - name: ColorJitter
          brightness: 0.4
          contrast: 0.4
          saturation: 0.4
          hue: 0.4
        - name: ToTensor
        - name: RandomErasing
          p: 0.25
        - name: Normalize
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
      COLLATE_FUNCTION: cutmixup_collator
      COLLATE_FUNCTION_PARAMS: {
        "ssl_method": "swav",
        "mixup_alpha": 1.0, # mixup alpha value, mixup is active if > 0.
        "cutmix_alpha": 1.0, # cutmix alpha value, cutmix is active if > 0.
        "prob": 1.0, # probability of applying mixup or cutmix per batch or element
        "switch_prob": 0.5, # probability of switching to cutmix instead of mixup when both are active
        "mode": "batch", # how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        "correct_lam": True, # apply lambda correction when cutmix bbox clipped by image borders
        "label_smoothing": 0.1, # apply label smoothing to the mixed target tensor
        "num_classes": 1 # number of classes for target
      }

DeiT uses `RandAugment <https://arxiv.org/abs/1909.13719>`_, `Random Erasing <https://arxiv.org/abs/1708.04896>`_, `MixUp <https://arxiv.org/abs/1710.09412>`_, `CutMix <https://arxiv.org/abs/1905.04899>`_, and Label Smoothing. Note that MixUp, CutMix, and Label Smoothing are not implemented as VISSL transforms, but instead as a custom collator ``DATA.TRAIN.COLLATE_FUNCTION: cutmixup_collator``, and using Label Smoothing requires setting ``DATA.TRAIN.LABEL_TYPE: "zero"`` (see ``vissl/config/defaults.yaml`` for details).

The ``LOSS`` section contains the parameters for the SwAV loss (See :ref:`here<Train SwAV model>` for detailed information about how to use SwAV):

.. code-block:: yaml

  LOSS:
    name: swav_loss
    swav_loss:
      temperature: 0.1
      use_double_precision: False
      normalize_last_layer: True
      num_iters: 3
      epsilon: 0.05
      crops_for_assign: [0, 1]
      queue:
        queue_length: 0
        start_iter: 0

ConViT
--------------------------------------------

`ConViT <https://arxiv.org/abs/2103.10697>`_ was designed with the goal of combining the expressivity of transformers with the sample-efficiency of the convolutional inductive bias. This is achieved by modifying the self-attention layers. In addition to the standard *N* self-attention heads in each layer, each self-attention head is paired with a *positional* attention head. The positional attention heads are similar to the standard self-attention heads, except their weights are initialized such that they perform convolution. The network then learns the convolutional kernel weights for the positional attention heads (in addition to all the other parameters that are normally learned in a transformer during training), as well as learning a gating parameter that controls the relative contribution of positional- vs. standard self-attention for each pair of heads. These *gated positional self-attention* (GPSA) heads allow the network to leverage the benefits of convolution without the rigid structure imposed by traditional convolutional architectures. Let's take a look at the ``MODEL`` section of ``configs/config/pretrain/vision_transformer/supervised/16_gpu_convit_b`` (a ConViT-B+ in the paper) to see how the ConViT differs from the ViT and DeiT:

.. code-block:: yaml

  MODEL:
    TRUNK:
      NAME: convit
      VISION_TRANSFORMERS:
        IMAGE_SIZE: 224
        PATCH_SIZE: 16
        NUM_LAYERS: 12
        NUM_HEADS: 16
        HIDDEN_DIM: 1024 # Hidden = 64 * NUM_HEADS
        MLP_DIM: 4096 # MLP dimension = 4 * HIDDEN_DIM
        CLASSIFIER: token
        DROPOUT_RATE: 0
        ATTENTION_DROPOUT_RATE: 0
        DROP_PATH_RATE: 0.1 # stochastic depth dropout probability
        QKV_BIAS: False # Bias for QKV in attention layers.
        QK_SCALE: False # Scale
      CONVIT:
        N_GPSA_LAYERS: 10 # Number of gated positional self-attention layers. Remaining layers are standard self-attention layers.
        CLASS_TOKEN_IN_LOCAL_LAYERS: False # Whether to add class token in GPSA layers. Recommended not to because it has been shown to lower performance.
        # Locality strength determines how much the positional attention is focused on the
        # patch of maximal attention. "Alpha" in the paper. Equivalent to
        # the temperature of positional attention softmax.
        LOCALITY_STRENGTH: 1.
        # Dimensionality of the relative positional embeddings * 1/3
        LOCALITY_DIM: 10
        # Whether to initialize the positional attention to be local
        # (equivalent to a convolution). Not much of a point in having GPSA if not True.
        USE_LOCAL_INIT: True
    HEAD:
      PARAMS: [
        ["mlp", {"dims": [1024, 1000]}],
      ] # No hidden layer in head

We use a ConViT trunk by specifying ``MODEL.TRUNK.NAME: convit``. The parameters that ConViT has in common with other vision transformer trunks, such as ``NUM_LAYERS`` are specified in ``MODEL.TRUNK.VISION_TRANSFORMERS``, just as with the ViT and DeiT. The ConViT-specific parameters are specified in ``MODEL.TRUNK.CONVIT``. ``N_GPSA_LAYERS`` specifies the number of GPSA layers. The remaining ``NUM_LAYERS - N_GPSA_LAYERS`` layers (in this case 12 - 10 = 2) will be standard self-attention layers. ``CLASS_TOKEN_IN_LOCAL_LAYERS`` controls whether to include the class token from the beginning, and thus in the GPSA layers, or to add it at the first self-attention layer after the GPSA layers. The ConViT authors found that including the class token in the GPSA layers was detrimental to performance. ``LOCALITY_STRENGTH`` controls the "narrowness" of the positional attention (see Figure 3 in the `paper <https://arxiv.org/abs/2103.10697>`_). The ConViT also features a single linear head, in contrast to the MLP head of the ViT and DeiT.

Additional information
--------------------------------------------

Other important factors related to training include:

- **Synchronized batch norm**: Vision transformers typically don't use batch norm, but many self-supervised learning methods obtain optimal performance when using heads that have batch norm. Ensure sync batch norm is set up properly if you're using batch norm and training on multiple GPUs. See the :ref:`Swav Documentation<Train SwAV model>` for a walk-through on sync batch norm.

- **Mixed precision**: Using mixed precision variables can reduce memory usage and afford larger batch sizes. See the :ref:`Swav Documentation<Train SwAV model>` for a walk-through on sync mixed precision training.

- **Data augmentations**: Read about :ref:`data augmentations in VISSL<Using Data Transforms>`; the :ref:`Swav Documentation<Train SwAV model>` has details about using multi-crop.

Pre-trained models
--------------------
Pre-trained models will eventually be available in `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_

Citations
---------

* **ViT**

.. code-block:: none

    @misc{dosovitskiy2020image,
          title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
          author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
          year={2020},
          eprint={2010.11929},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

* **DeiT**

.. code-block:: none

    @misc{touvron2021training,
          title={Training data-efficient image transformers & distillation through attention},
          author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Hervé Jégou},
          year={2021},
          eprint={2012.12877},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

* **ConViT**

.. code-block:: none

    @misc{dascoli2021convit,
          title={ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases},
          author={Stéphane d'Ascoli and Hugo Touvron and Matthew Leavitt and Ari Morcos and Giulio Biroli and Levent Sagun},
          year={2021},
          eprint={2103.10697},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
