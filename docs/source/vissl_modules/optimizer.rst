Using Optimizers
===============================

VISSL support all PyTorch optimizers (SGD, Adam etc) and `ClassyVision optimizers <https://github.com/facebookresearch/ClassyVision/tree/main/classy_vision/optim>`_.


Creating Optimizers
--------------------

The optimizers can be easily created from the configuration files. The user needs to set the optimizer name in :code:`OPTIMIZER.name`. Users can configure other settings like #epochs, etc as follows:

.. code-block:: yaml

    OPTIMIZER:
        name: sgd
        weight_decay: 0.0001
        momentum: 0.9
        nesterov: False
        # for how many epochs to do training. only counts training epochs.
        num_epochs: 90
        # whether to regularize batch norm. if set to False, weight decay of batch norm params is 0.
        regularize_bn: False
        # whether to regularize bias parameter. if set to False, weight decay of bias params is 0.
        regularize_bias: True

Using different LR for Head and trunk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL supports using a different LR and weight decay for head and trunk. User needs to set the config option :code:`OPTIMIZER.head_optimizer_params.use_different_values=True` in order to enable
this functionality.

.. code-block:: yaml

    OPTIMIZER:
      head_optimizer_params:
        # if the head should use a different LR than the trunk. If yes, then specify the
        # param_schedulers.lr_head settings. Otherwise if set to False, the
        # param_scheduelrs.lr will be used automatically.
        use_different_lr: False
        # if the head should use a different weight decay value than the trunk.
        use_different_wd: False
        # if using different weight decay value for the head, set here. otherwise, the
        # same value as trunk will be automatically used.
        weight_decay: 0.0001

Using LARC
~~~~~~~~~~~~~~

VISSL supports the LARC implementation from `NVIDIA's Apex LARC <https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py>`_. To use LARC, users need to set config option
:code:`OPTIMIZER.use_larc=True`. VISSL exposes LARC parameters that users can tune. Full list of LARC parameters exposed by VISSL:

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


Creating LR Schedulers
--------------------------

Users can use different types of Learning rate schedules for the training of their models. We closely follow the `LR schedulers supported by ClassyVision <https://github.com/facebookresearch/ClassyVision/tree/main/classy_vision/optim/param_scheduler>`_ and also custom
`learning rate schedules in VISSL <https://github.com/facebookresearch/vissl/tree/main/vissl/optimizers/param_scheduler>`_.

How to set learning rate
~~~~~~~~~~~~~~~~~~~~~~~~~~

Below we provide some examples of how to setup various types of Learning rate schedules. Note that these are merely some examples and you should set your desired parameter values.

1. Cosine

.. code-block:: yaml

    OPTIMIZER:
      param_schedulers:
        lr:
          name: cosine
          start_value: 0.15   # LR for batch size 256
          end_value: 0.0000


2. Multi-Step


.. code-block:: yaml

    OPTIMIZER:
      param_schedulers:
        lr:
          name: multistep
          values: [0.01, 0.001]
          milestones: [1]
          update_interval: epoch  # update LR after every epoch


3. Linear Warmup + Cosine


.. code-block:: yaml

    OPTIMIZER:
      param_schedulers:
        lr:
          name: composite
          schedulers:
            - name: linear
                start_value: 0.6
                end_value: 4.8
            - name: cosine
                start_value: 4.8
                end_value: 0.0048
          interval_scaling: [rescaled, fixed]
          update_interval: step
          lengths: [0.1, 0.9]                 # 100ep


4. Cosine with restarts

.. code-block:: yaml

    OPTIMIZER:
      param_schedulers:
        lr:
          name: cosine_warm_restart
          start_value: 0.15   # LR for batch size 256
          end_value: 0.00015
          restart_interval_length: 0.5
          wave_type: half  # full | half


5. Linear warmup + cosine with restarts

.. code-block:: yaml

    OPTIMIZER:
      param_schedulers:
        lr:
          name: composite
          schedulers:
            - name: linear
                start_value: 0.6
                end_value: 4.8
            - name: cosine_warm_restart
                start_value: 4.8
                end_value: 0.0048
                # wave_type: half
                # restart_interval_length: 0.5
                wave_type: full
                restart_interval_length: 0.334
          interval_scaling: [rescaled, rescaled]
          update_interval: step
          lengths: [0.1, 0.9]                 # 100ep


6. Multiple linear warmups and cosine

.. code-block:: yaml

    OPTIMIZER:
      param_schedulers:
        lr:
          schedulers:
            - name: linear
                start_value: 0.6
                end_value: 4.8
            - name: cosine
                start_value: 4.8
                end_value: 0.0048
            - name: linear
                start_value: 0.0048
                end_value: 2.114
            - name: cosine
                start_value: 2.114
                end_value: 0.0048
          update_interval: step
          interval_scaling: [rescaled, rescaled, rescaled, rescaled]
          lengths: [0.0256, 0.48722, 0.0256, 0.46166]         # 1ep IG-500M



Auto-scaling of Learning Rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL supports automatically scaling LR as per `Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour <https://arxiv.org/abs/1706.02677>`_.
To turn this automatic scaling on, set :code:`config.OPTIMIZER.param_schedulers.lr.auto_lr_scaling.auto_scale=true`.

:code:`scaled_lr` is calculated: for a given

- :code:`base_lr_batch_size` = batch size for which the base learning rate is specified.

- :code:`base_value` = base learning rate value that will be scaled, the current batch size is used to determine how to scale the base learning rate value.

:code:`scale_factor = (batchsize_per_gpu * world_size) / base_lr_batch_size`

if :code:`scaling_type` is set to "sqrt", :code:`scale_factor = sqrt(scale_factor)`

:code:`scaled_lr = scale_factor * base_value`

For different types of learning rate schedules, the LR scaling is handled as below:

.. code-block:: bash

    1. cosine:
        end_value = scaled_lr * (end_value / start_value)
        start_value = scaled_lr and
    2. multistep:
        gamma = values[1] / values[0]
        values = [scaled_lr * pow(gamma, idx) for idx in range(len(values))]
    3. step_with_fixed_gamma
        base_value = scaled_lr
    4. linear:
       end_value = scaled_lr
    5. inverse_sqrt:
       start_value = scaled_lr
    6. constant:
       value = scaled_lr
    7. composite:
        recursively call to scale each composition. If the composition consists of a linear
        schedule, we assume that a linear warmup is applied. If the linear warmup is
        applied, it's possible the warmup is not necessary if the global batch_size is smaller
        than the base_lr_batch_size and in that case, we remove the linear warmup from the
        schedule.

Here is an example configuration for linear scaling, with a base batchsize of 256, and a base learning rate of 0.1:

.. code-block:: yaml

    OPTIMIZER:
      param_schedulers:
         lr:
           # we make it convenient to scale Learning rate automatically as per the scaling
           # rule specified in https://arxiv.org/abs/1706.02677 (ImageNet in 1Hour).
           auto_lr_scaling:
             # if set to True, learning rate will be scaled.
             auto_scale: True
             # base learning rate value that will be scaled.
             base_value: 0.1
             # batch size for which the base learning rate is specified. The current batch size
             # is used to determine how to scale the base learning rate value.
             # scaled_lr = ((batchsize_per_gpu * world_size) * base_value ) / base_lr_batch_size
             base_lr_batch_size: 256
             # scaling_type can be set to "sqrt" to reduce the impact of scaling on the base value
             scaling_type: "linear"
