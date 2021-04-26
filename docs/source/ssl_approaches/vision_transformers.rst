Working with Vision Transformers in vissl
============================================

Author: matthew.l.leavitt@gmail.com

VISSL contains implementations of multiple vision transformer model variants:

- **Vision Transformers (ViT)**: Published in |vision_transformer_cite|_, ViT was a breakthrough for transformers in vision tasks, and comprises a set of model architectures and hyperparameters that closely follow `Vaswani et al.'s original implementation of transformers for NLP <https://arxiv.org/abs/1706.03762>`_.

- **Data-efficient image Transformers (DeiT)**: Published in |deit_cite|_, DeiT is architecturally similar to ViT, but is distinguished by its hyperparameters and use of distillation during training. Training with distillation is not currently supported in VISSL, but DeiT provides benefits even when training without distillation.

- **Convolutional Vision Transformer (ConViT)**: Published in |convit_cite|_, the ConViT was designed with the goal of combining the expressivity of transformers with the sample-efficiency of the convolutional inductive bias. ConViT achieves this by replacing the standard self-attention layers of the vision transformer with "gated positional self-attention" layers that are initialized to perform both convolution and self-attention, and learn the optimal trade-off between the two functions.

.. |vision_transformer_cite| replace:: Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020)
.. _vision_transformer_cite: https://arxiv.org/abs/2010.11929

.. |deit_cite| replace:: Touvron et al., *Training data-efficient image transformers & distillation through attention* (2020)
.. _deit_cite: https://arxiv.org/abs/2012.12877

.. |convit_cite| replace:: d'Ascoli et al., *ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases* (2021)
.. _convit_cite: https://arxiv.org/abs/2103.10697

We start by demonstrating how to train ViTs, and continue with examples of training DeiTs and ConViTs.

How to train a vision transformer
----------------------------------

VISSL provides yaml configuration files containing the exact hyperparam settings to reproduce the model. VISSL implements
all the components including loss, data augmentations, collators etc required for this approach.

To train ResNet-50 model on 8-gpus on ImageNet-1K dataset using feature projection dimension 128 for memory:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/swav/swav_8node_resnet

Example yaml code
--------------------------------------------

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

DeiT
--------------------------------


ConViT
--------------------------------------------



Stuff from other docs applies here too
--------------------------------------------

Mixed precision. Sync batch norm

Gradient clipping
--------------------------------------------


Augmentation settings
--------------------------------------------

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
Pre-trained models will eventually be available in `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/master/MODEL_ZOO.md>`_


Citations
---------

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