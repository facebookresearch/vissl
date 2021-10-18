Benchmark task: Full-finetuning
===========================================================

Using a self-supervised model to initialize a network and finetune the full weights on the target task is a very common evaluation protocol.
This benchmark requires only initializing the model -- no other settings in :code:`MODEL.FEATURE_EVAL_SETTINGS` are needed unlike other benchmark tasks.

Benchmark: ImageNet-1k
---------------------------

The configuration for full fine-tuning on Imagenet is available in `benchmark/fulltune/imagenet1k <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/fulltune/imagenet1k>`_ and can be run as follows:


.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/fulltune/imagenet1k/eval_resnet_8gpu_transfer_in1k_fulltune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

Configurations for fine-tuning on a sub-set of Imagenet are also available and can be run as follows:

.. code-block::

    # For fine-tuning on 1% of the dataset
    python tools/run_distributed_engines.py \
      config=benchmark/fulltune/imagenet1k/eval_resnet_8gpu_transfer_in1k_fulltune \
      +config/benchmark/fulltune/imagenet1k/dataset=imagenet1k_1percent \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

    # For fine-tuning on 10% of the dataset
    python tools/run_distributed_engines.py \
      config=benchmark/fulltune/imagenet1k/eval_resnet_8gpu_transfer_in1k_fulltune \
      +config/benchmark/fulltune/imagenet1k/dataset=imagenet1k_10percent \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: Places205
---------------------------

The configuration for full fine-tuning on Places205 is available in `benchmark/fulltune/places205 <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/fulltune/places205>`_ and can be run as follows:


.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/fulltune/places205/eval_resnet_8gpu_transfer_places205_fulltune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>



.. note::

    Please see VISSL documentation on how to run a given training on **1-gpu, multi-gpu or multi-machine**.

.. note::

    Please see VISSL documentation on how to use the **builtin datasets** if you want to run this benchmark on a different target task..

.. note::

    Please see VISSL documentation on how to use YAML comfiguration system in VISSL to **override specific components like model** of a config file. For example,
    in the above file, user can replace ResNet-50 model with a different architecture like RegNetY-256 etc. easily.
