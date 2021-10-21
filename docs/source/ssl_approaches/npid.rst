Train NPID (and NPID++) model
===============================

VISSL reproduces the self-supervised approach **Unsupervised Feature Learning via Non-Parametric Instance Discrimination**
proposed by **Zhirong Wu, Yuanjun Xiong, Stella Yu, Dahua Lin** in `this paper <https://arxiv.org/abs/1805.01978>`_. The NPID baselines were improved
further by **Misra et. al** in **Self-Supervised Learning of Pretext-Invariant Representations** proposed in `this paper <https://arxiv.org/abs/1912.01991>`_.

How to train NPID model
---------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including loss, data augmentations, collators etc required for this approach.

To train ResNet-50 model on 8-gpus on ImageNet-1K dataset with NPID approach using 4,096 negatives selected randomly and feature projection dimension 128:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/npid/npid_8gpu_resnet


How to Train NPID++ model
-------------------------------

To train the NPID++ baselines with a ResNet-50 on ImageNet with 32000 negatives, 800 epochs and 4 machines (32-gpus) as in the PIRL paper:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/npid/npid++_4nodes_resnet


Vary the training loss settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VISSL supports many backbone architectures including AlexNet, ResNe(X)ts. Some examples below:


* **Train ResNet-101:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/npid/npid_8gpu_resnet \
        config.MODEL.TRUNK.NAME=resnet config.MODEL.TRUNK.RESNETS.DEPTH=101


Vary the number of gpus
~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the NPID model on 4 machines (32gpus)
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/npid/npid_8gpu_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1


* **Training on 4 machines i.e. 32-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/npid/npid_8gpu_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=4


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
VISSL using NPID and NPID++ approach and the benchmarks.


Citations
---------

* **NPID**

.. code-block:: none

    @misc{wu2018unsupervised,
        title={Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination},
        author={Zhirong Wu and Yuanjun Xiong and Stella Yu and Dahua Lin},
        year={2018},
        eprint={1805.01978},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }


* **NPID++**

.. code-block:: none

    @misc{misra2019selfsupervised,
        title={Self-Supervised Learning of Pretext-Invariant Representations},
        author={Ishan Misra and Laurens van der Maaten},
        year={2019},
        eprint={1912.01991},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
