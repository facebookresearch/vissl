Train MoCo model
===============================

Author: lefaudeux@fb.com

VISSL reproduces the self-supervised approach MoCo **Momentum Contrast for Unsupervised Visual Representation Learning**
proposed by **Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick** in `this paper <https://arxiv.org/abs/1911.05722>`_. The MoCo baselines were improved
further by **Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He** in "Improved Baselines with Momentum Contrastive Learning" proposed in `this paper <https://arxiv.org/abs/2003.04297>`_.

VISSL closely follows the `implementation <https://github.com/facebookresearch/moco>`_ provided by MoCo authors themselves.

How to train MoCo (and MoCo v2 model) model
--------------------------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including loss, data augmentations, collators etc required for this approach.

To train ResNet-50 model on 8-gpus on ImageNet-1K dataset with MoCo-v2 approach using feature projection dimension 128:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/moco/moco_1node_resnet


By default, VISSL provides configuration file for MoCo-v2 model as this has better baselines numbers. To train MoCo baseline instead,
users should make 2 changes to the moco configuration file:

- change the :code:`config.DATA.TRAIN.TRANSFORMS` by removing the :code:`ImgPilGaussianBlur` transform.
- change the :code:`config.MODEL.HEAD.PARAMS=[["mlp", {"dims": [2048, 128]}]]` i.e. replace the MLP-head with fc-head.


Vary the training loss settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Users can adjust several settings from command line to train the model with different hyperparams. For example: to use a different momentum value (say 0.99) for memory and different
temperature 0.5 for logits, the MoCo training command would look like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/moco/moco_1node_resnet \
        config.LOSS.moco_loss.temperature=0.5 \
        config.LOSS.moco_loss.momentum=0.99

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    moco_loss:
      embedding_dim: 128
      queue_size: 65536
      momentum: 0.999
      temperature: 0.2

Training different model architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VISSL supports many backbone architectures including ResNe(X)ts, wider ResNets. Some examples below:


* **Train ResNet-101:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/moco/moco_1node_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101

* **Train ResNet-50-w2 (2x wider ResNet-50):**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/moco/moco_1node_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=50 \
        config.MODEL.TRUNK.RESNETS.WIDTH_MULTIPLIER=2


Vary the number of gpus
~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the MoCo model on 4 machines (32-gpus)
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/moco/moco_1node_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1


* **Training on 4 machines i.e. 32-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/moco/moco_1node_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=4


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.
    However, MoCo doesn't work very well with this rule as per the authors in the paper.

.. note::

    If you change the number of gpus for MoCo training, MoCo models require longer training in order to reproduce results.
    Hence, we recommend users to consult MoCo paper.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
VISSL using MoCo-v2 approach and the benchmarks.


Citations
---------

* **MoCo**

.. code-block:: none

    @misc{he2020momentum,
        title={Momentum Contrast for Unsupervised Visual Representation Learning},
        author={Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
        year={2020},
        eprint={1911.05722},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }


* **MoCo-v2**

.. code-block:: none

    @misc{chen2020improved,
        title={Improved Baselines with Momentum Contrastive Learning},
        author={Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
        year={2020},
        eprint={2003.04297},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
