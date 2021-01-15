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

VISSL uses `FAIRScale <https://github.com/facebookresearch/fairscale>`_ library which implements ZeRO in PyTorch.
Using VISSL in ZeRO involves no code changes and can simply be done by setting some configuration options in the yaml files.

In order to use ZeRO, user needs to set :code:`OPTIMIZER.name=zero` and nest the desired optimizer (for example SGD) settings in :code:`OPTIMIZER.base_optimizer`.

An example for using ZeRO with LARC and SGD optimization:

.. code-block:: yaml

    OPTIMIZER:
      name: zero
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
