Activation checkpointing to reduce model memory
==================================================

Authors: m1n@fb.com, lefaudeux@fb.com

Activation checkpointing is a very powerful technique to reduce the memory requirement of a model. This is especially useful when training very large models with billions of parameters.

How it works?
---------------

Activation checkpointing trades compute for memory. It discards intermediate activations during the forward pass, and recomputes them during the backward pass. In
our experiments, using activation checkpointing, we observe negligible compute overhead in memory-bound settings while getting big memory savings.

In summary, This technique offers 2 benefits:

- saves gpu memory that can be used to fit large models
- allows increasing training batch size for a given model

We recommend users to read the documentation available `here <https://pytorch.org/docs/stable/checkpoint.html>`_ for further details on activation checkpointing.

How to use activation checkpointing in VISSL?
----------------------------------------------

VISSL integrates activation checkpointing implementation directly from PyTorch available `here <https://pytorch.org/docs/stable/checkpoint.html>`_.
Using activation checkpointing in VISSL is extremely easy and doable with simple settings in the configuration file. The settings required are as below:

.. code-block:: yaml

    MODEL:
      ACTIVATION_CHECKPOINTING:
        # whether to use activation checkpointing or not
        USE_ACTIVATION_CHECKPOINTING: True
        # how many times the model should be checkpointed. User should tune this parameter
        # and find the number that offers best memory saving and compute tradeoff.
        NUM_ACTIVATION_CHECKPOINTING_SPLITS: 8
    DISTRIBUTED:
      # if True, does the gradient reduction in DDP manually. This is useful during the
      # activation checkpointing and sometimes saving the memory from the pytorch gradient
      # buckets.
      MANUAL_GRADIENT_REDUCTION: True
