Benchmark task: Full finetuning on Imagenet 1% , 10% subsets
====================================================================

Evaluating a self-supervised pre-trained model on the target dataset which represents 1% or 10% of Imagenet dataset has become a very common evaluation criterion. VISSL provides
the benchmark settings for this benchmark `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/semi_supervised/imagenet1k>`_.


1% and 10% Data subsets
-------------------------

VISSL uses the 1% and 10% datasets from the SimCLR work. Users can `download the datasets from here <https://github.com/google-research/simclr/tree/master/imagenet_subsets>`_.
Users can use the :code:`DATA.TRAIN.DATA_SOURCES=[disk_filelist]` to load the images in these files. Users should replace each line with the valid full image path for that image.
Users should also extract the labels out from these datasets using the :code:`image_id` of each image.

Once the user has the valid image and labels files (.npy), users should set the dataset paths in
`VISSL dataset_catalog.json <https://github.com/facebookresearch/vissl/blob/main/configs/config/dataset_catalog.json>`_ for the datasets
:code:`google-imagenet1k-per01` and :code:`google-imagenet1k-per10`


Benchmark: 1% ImageNet
-------------------------

Users can run the benchmark on 1% ImageNet subsets from SimCLR with the following command:

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/semi_supervised/imagenet1k/eval_resnet_8gpu_transfer_in1k_semi_sup_fulltune \
      +config/benchmark/semi_supervised/imagenet1k/dataset=simclr_in1k_per01


Benchmark: 10% ImageNet
-------------------------

Users can run the benchmark on 10% ImageNet subsets from SimCLR with the following command:

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/semi_supervised/imagenet1k/eval_resnet_8gpu_transfer_in1k_semi_sup_fulltune \
      +config/benchmark/semi_supervised/imagenet1k/dataset=simclr_in1k_per10


.. note::

    Please see VISSL documentation on how to run a given training on **1-gpu, multi-gpu or multi-machine**.

.. note::

    Please see VISSL documentation on how to use the **builtin datasets** if you want to run this benchmark on a different target task..

.. note::

    Please see VISSL documentation on how to use YAML comfiguration system in VISSL to **override specific components like model** of a config file. For example,
    in the above file, user can replace ResNet-50 model with a different architecture like RegNetY-256 etc. easily.
