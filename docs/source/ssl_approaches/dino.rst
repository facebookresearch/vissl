Train DINO model
===============================

Author: mathilde@fb.com

VISSL reproduces the self-supervised approach called :code:`DINO` presented in **Emerging Properties in Self-Supervised Vision Transformers** which was proposed by
**Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, Julien Mairal, Piotr Bojanowski and Armand Joulin** in `this paper <https://arxiv.org/abs/2104.14294>`_.

How to train DINO model
----------------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including loss, data augmentations, collators etc required for this approach.

To train DeiT-S/16 model on 16-gpus on ImageNet-1K dataset:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/dino/dino_16gpus_deits16

Vary the training loss settings
---------------------------------
Users can adjust several settings from command line to train the model with different hyperparams. For example: to use a different
temperature 0.2 for the student, the training command would look like:

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/dino/dino_16gpus_deits16 \
        config.LOSS.dino_loss.student_temp=0.2

The full set of loss params that VISSL allows modifying:

.. code-block:: yaml

    dino_loss:
      momentum: 0.996 # base momentum parameter used for teacher model
      student_temp: 0.1 # student temperature
      teacher_temp_min: 0.04 # warmup teacher temperature
      teacher_temp_max: 0.07 # base teacher temperature
      teacher_temp_warmup_iters: 37500 # 30 epochs
      crops_for_teacher: [0, 1] # crops used by the teacher
      ema_center: 0.9 # momentum parameter used for updating the teacher center
      normalize_last_layer: true # should we l2-normalize the last layer
      output_dim: 65536  # automatically inferred from model HEAD settings
