ZeRO: Optimizer state and gradient sharding
==============================================

Author: lefaudeux@fb.com

**ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** is a technique developed by **Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He** in `this paper <https://arxiv.org/abs/1910.02054>`_.
When training models with billions of parameters, GPU memory becomes a bottleneck. ZeRO can offer 4x to 8x memory reductions in memory thus allowing
to fit larger models in memory.

How ZeRO works?
------------------

Memory requirement of a model can be broken down roughly into:

1. activations memory
2. model parameters
3. parameters momentum buffers (optimizer state)
4. parameters gradients

ZeRO *shards* the optimizer state and the parameter gradients onto different devices and reduces the memory needed per device.

How to use ZeRO in VISSL?
--------------------------

VISSL uses `FAIRScale <https://github.com/facebookresearch/fairscale>`_ library which implements ZeRO in PyTorch. To use Zero, you can either use Sharded Data Parallel (SDP), inspired by ZeRO-2 or Fully Sharded Data Parallel, which was inspired by ZeRO-3. The main difference between the two, is that SDP shards the gradients and optimizer state, whereas FSDP additionally shards the model parameters. This decreases memory at the expense of communication. For more information see `this FAIRscale doc <https://fairscale.readthedocs.io/en/latest/deep_dive/oss_sdp_fsdp.html>`_.

Using VISSL in ZeRO involves no code changes and can simply be done by setting some configuration options in the yaml files.

In order to use ZeRO, user needs to set :code:`OPTIMIZER.name=zero` and nest the desired optimizer (for example SGD) settings in :code:`OPTIMIZER.base_optimizer`.

An example for using ZeRO with LARC and SGD optimization:

.. code-block:: yaml

    OPTIMIZER:
      name: "zero"
      use_zero: True
      base_optimizer:
        name: sgd
        use_larc: False
        larc_config:
          clip: False
          trust_coefficient: 0.001
          eps: 0.00000001
        weight_decay: 0.000001
        momentum: 0.9
        nesterov: False
.. note::

    ZeRO works seamlessly with LARC and mixed precision training. Using ZeRO with activation checkpointing is not yet enabled primarily due to manual gradient reduction need for activation checkpointing.

To use Sharded Data Parallel (SDP), inspired by Zero-2, merely set:

.. code-block:: yaml

    MODEL:
      SHARDED_DDP_SETUP:
         # set this to true if you want to use SDP instead of DDP.
         # VISSL will automatically set optimizer = zero and
         # configure the settings required to run SDP successfully.
         USE_SDP: True
         reduce_buffer_size: -1

To use Fully Sharded Data Parallel (FSDP), inspired by Zero-3, merely set:

.. code-block:: yaml

    MODEL:
       FSDP_CONFIG:
         # set this option to True to enable FSDP and automatically determine the config
         # for FSDP based on AMP true/false.
         AUTO_SETUP_FSDP: True
         # Set this option to a positive number to automatically wrap "big" layers with
         # a dedicated FSDP wrapping: the number provided here is the number of
         # parameters that serves as threshold to decide if a layer is "big"
         AUTO_WRAP_THRESHOLD: 0
         AMP_TYPE: "01"
         # Parameters of fairscale FSDP
         flatten_parameters: True
         mixed_precision: True
         fp32_reduce_scatter: False  # Only makes sense to be True when mixed_precision is True.
         compute_dtype: float32  # Choose "float32" or "float16"
         bucket_cap_mb: 0
         clear_autocast_cache: True
         verbose: True

**Warning:** This has only been fully tested with SwAV + Regnet models.
