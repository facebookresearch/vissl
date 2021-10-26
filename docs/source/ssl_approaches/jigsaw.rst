Train Jigsaw model
===============================

VISSL reproduces the self-supervised approach **Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles**
proposed by **Mehdi Noroozi and Paolo Favaro** in `this paper <https://arxiv.org/abs/1603.09246>`_.

How to train Jigsaw model
---------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including data augmentations, collators etc required for this approach.

To train ResNet-50 model on 8-gpus on ImageNet-1K dataset using 2000 permutations.

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/jigsaw/jigsaw_8gpu_resnet


Training with different permutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In order to adjust the permutations and retrain, you can do so from the command line. For example: to train for 10K permutations instead,
VISSL provides the configuration files with the necessary changes. Run:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/jigsaw/jigsaw_8gpu_resnet \
        +config/pretrain/jigsaw/permutations=perm10K

Similarly, you can train for 100 permutations and create new config files for a different permutations settings following the above configs
as examples.


Vary the number of gpus
~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the Jigsaw model on 4 machines (32gpus)
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/jigsaw/jigsaw_8gpu_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1


* **Training on 4 machines i.e. 32-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/jigsaw/jigsaw_8gpu_resnet \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=4


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
VISSL using Jigsaw approach and the benchmarks.


Permutations
--------------
Following `Goyal et al <https://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf>`_
we use the exact permutation files for Jigsaw training `available here <https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#jigsaw-permutations>`_ and refer
users to directly use the files from the above source.


Citation
---------

.. code-block:: none

    @misc{noroozi2017unsupervised,
        title={Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles},
        author={Mehdi Noroozi and Paolo Favaro},
        year={2017},
        eprint={1603.09246},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
