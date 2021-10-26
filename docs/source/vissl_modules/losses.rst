Using PyTorch and VISSL Losses
===============================


VISSL supports all PyTorch loss functions and also implements several loss functions that are specific to self-supervised approaches like DINO, SwAV, MoCo, PIRL, SimCLR, etc. Using any loss is very easy in VISSL and involves simply editing the configuration files to specify the loss name
and the parameters of that loss. See all the `VISSL custom losses here <https://github.com/facebookresearch/vissl/tree/main/vissl/losses>`_.

To use a certain loss, users need to simply set :code:`LOSS.name=<my_loss_name>` and set the parameter values that loss requires.


Using Cross Entropy Loss for Training and Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    LOSS:
      name: "CrossEntropyLoss"
      # ----------------------------------------------------------------------------------- #
      # Standard PyTorch Cross-Entropy Loss. Use the loss name exactly as in PyTorch.
      # pass any variables that the loss takes.
      # ----------------------------------------------------------------------------------- #
      CrossEntropyLoss:
        ignore_index: -1


Using SwAV loss for Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        temp_hard_assignment_iters: 0
        num_crops: 2                  # automatically inferred from data transforms
        num_prototypes: [3000]        # automatically inferred from model HEAD settings
        embedding_dim: 128            # automatically inferred from HEAD params
        # for dumping the debugging info in case loss becomes NaN
        output_dir: ""                # automatically inferred and set to checkpoint dir
        queue:
          local_queue_length: 0       # automatically inferred to queue_length // world_size
          queue_length: 0             # automatically adjusted to ensure queue_length % global batch size = 0
          start_iter: 0

Using DINO loss for Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    LOSS:
      name: dino_lodd
      dino_loss:
        momentum: 0.996
        student_temp: 0.1
        teacher_temp_min: 0.04
        teacher_temp_max: 0.07
        teacher_temp_warmup_iters: 37500 # 30 epochs
        crops_for_teacher: [0, 1]
        ema_center: 0.9
        normalize_last_layer: true
        output_dim: 65536  # automatically inferred from model HEAD settings
