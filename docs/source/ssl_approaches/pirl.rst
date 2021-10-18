Train PIRL model
===============================

Author: imisra@fb.com

VISSL reproduces the self-supervised approach **Self-Supervised Learning of Pretext-Invariant Representations** proposed by **Ishan Misra and Laurens van der Maaten** in `this paper <https://arxiv.org/abs/1912.01991>`_.

How to train PIRL model
---------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including loss, data augmentations, collators etc required for this approach.

To train ResNet-50 model on 4-machines (8-nodes) on ImageNet-1K dataset with PIRL approach using 32,000 negatives selected randomly and feature projection dimension 128:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50


Vary the training loss settings
------------------------------------------------
Users can adjust several settings from command line to train the model with different hyperparams. For example: to use a different momentum value (say 0.99) for memory and different
temperature 0.05 for logits, using 16000 negatives, the NPID training command would look like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/npid/npid_8gpu_resnet \
        config.LOSS.nce_loss_with_memory.temperature=0.05 \
        config.LOSS.nce_loss_with_memory.memory_params.momentum=0.99 \
        config.LOSS.nce_loss_with_memory.negative_sampling_params.num_negatives=16000

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    nce_loss_with_memory:
      # setting below to "cross_entropy" yields the InfoNCE loss
      loss_type: "nce"
      norm_embedding: True
      temperature: 0.07
      # if the NCE loss is computed between multiple pairs, we can set a loss weight per term
      # can be used to weight different pair contributions differently.
      loss_weights: [1.0]
      norm_constant: -1
      update_mem_with_emb_index: -100
      negative_sampling_params:
        num_negatives: 16000
        type: "random"
      memory_params:
        memory_size: -1
        embedding_dim: 128
        momentum: 0.5
        norm_init: True
        update_mem_on_forward: True
      # following parameters are auto-filled before the loss is created.
      num_train_samples: -1    # @auto-filled


Training different model architecture
------------------------------------------------
VISSL supports many backbone architectures including ResNe(X)ts, wider ResNets. Some examples below:


* **Train ResNet-101:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101

* **Train ResNet-50-w2 (2x wider):**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101 \
        config.MODEL.TRUNK.RESNETS.WIDTH_MULTIPLIER=2


Training with Gaussian Blur augmentation
------------------------------------------------

Gaussian Blur augmentation has being a crucial transformation for better performance in approaches like
SimCLR, SwAV, etc. The original PIRL method didn't use Gaussian Blur augmentation however PIRL author (imisra@fb.com)
provide configuration for how to use the Gaussian Blur for training PIRL models. The command to run:


.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
        +config/pretrain/pirl/transforms=photo_gblur

Please consult the `photo_gblur.yaml` config for the transformation composition.

Training with MLP head
------------------------------------------------

Recent self-supervised approaches like SimCLR, MoCo, SwAV have benefitted significantly from using an MLP
head. Original PIRL work didn't use MLP head but PIRL author (imisra@fb.com) provide configuration for using
MLP head in PIRL and also open source the models. The command to run:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
        +config/pretrain/pirl/models=resnet50_mlphead

Similarly, to train a ResNet-50-w2 (ie. 2x wider ResNet-50) with PIRL using MLP head:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
        +config/pretrain/pirl/models=resnet50_w2_mlphead

Similarly, to train a ResNet-50-w4 (ie. 4x wider ResNet-50) with PIRL using MLP head:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
        +config/pretrain/pirl/models=resnet50_w4_mlphead


Vary the number of epochs
------------------------------------------------

In order to vary the number of epochs to use for training PIRL models, one can achieve this simply
from command line. For example, to train the PIRL model for 100 epochs instead, pass the `num_epochs`
parameter from command line:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
        config.OPTIMIZER.num_epochs=100


Vary the number of gpus
------------------------------------------------

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the PIRL model on 8-gpus
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1 config.DISTRIBUTED.NUM_NODES=1


* **Training on 8-gpus:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=1


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
VISSL for PIRL and the benchmarks.


Citations
---------

.. code-block:: none

    @misc{misra2019selfsupervised,
        title={Self-Supervised Learning of Pretext-Invariant Representations},
        author={Ishan Misra and Laurens van der Maaten},
        year={2019},
        eprint={1912.01991},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
