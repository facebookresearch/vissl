Benchmark task: Full-finetuning
===========================================================

Using a self-supervised model to initialize a network and further tune the weights on the target task is a very common evaluation protocol. This benchmark requires only initializing the model and no other settings in :code:`MODEL.FEATURE_EVAL_SETTINGS` are needed unlike other benchmark tasks.

Benchmark: ImageNet-1k
---------------------------
VISSL provides the YAML configuration setting for this benchmark `here <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/imagenet1k_fulltune>`_ which can be run as below.


.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/imagenet1k_fulltune/eval_resnet_8gpu_transfer_in1k_fulltune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: Places205
---------------------------
VISSL provides the YAML configuration setting for this benchmark `here <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/places205_fulltune>`_ which can be run as below.


.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/places205_fulltune/eval_resnet_8gpu_transfer_places205_fulltune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>



.. note::

    Please see VISSL documentation on how to run a given training on **1-gpu, multi-gpu or multi-machine**.

.. note::

    Please see VISSL documentation on how to use the **builtin datasets** if you want to run this benchmark on a different target task..

.. note::

    Please see VISSL documentation on how to use YAML comfiguration system in VISSL to **override specific components like model** of a config file. For example,
    in the above file, user can replace ResNet-50 model with a different architecture like RegNetY-256 etc. easily.
