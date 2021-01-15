Benchmark: Linear Image Classification
===========================================================

VISSL provides a standardized benchmark suite to evaluate the feature representation quality of self-supervised pretrained models. A popular
evaluation protocol is to freeze the model traink and train linear classifiers on several layers of the model on some target datasets (like ImageNet-1k, Places205, VOC07, iNaturalist2018).
In VISSL, we support all the linear evals on all the datasets. We also provide standard set of hyperparams for various approaches
in order to reproduce the model performance in SSL literature. For reproducibility, see `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/master/MODEL_ZOO.md>`_.

.. note::

    To run the benchmark, we recommend using the standard set of hyperparams provided by VISSL as these hyperparams reproduce results of large number of self-supervised approaches.
    Users are however free to modify the hyperparams to suit their evaluation criterion.

Eval Config settings using MLP head
--------------------------------------

Set the following in the config file to enable the feature evaluation

Attaching MLP head to many layers of trunk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- To attach linear classifier (:code:`FC`) on the trunk output, example for a ResNet-50 model:

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_ONLY: True
        SHOULD_FLATTEN_FEATS: False
        LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
            ["res5avg", ["Identity", []]],
        ]
      TRUNK:
        NAME: resnet
        TRUNK_PARAMS:
        RESNETS:
            DEPTH: 50
      HEAD:
        PARAMS: [
            ["mlp", {"dims": [2048, 1000]}],
        ]


Attaching MLP head to trunk output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To attach a linear classifier at multiple layers of model following Zhang et. al style which has :code:`BN -> FC` as the head, use :code:`eval_mlp` head. For example, for a ResNet-50 model,

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_ONLY: True
        SHOULD_FLATTEN_FEATS: False
        LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
          ["conv1", ["AvgPool2d", [[10, 10], 10, 4]]],
          ["res2", ["AvgPool2d", [[16, 16], 8, 0]]],
          ["res3", ["AvgPool2d", [[13, 13], 5, 0]]],
          ["res4", ["AvgPool2d", [[8, 8], 3, 0]]],
          ["res5", ["AvgPool2d", [[6, 6], 1, 0]]],
          ["res5avg", ["Identity", []]],
        ]
      TRUNK:
        NAME: resnet
        TRUNK_PARAMS:
        RESNETS:
            DEPTH: 50
      HEAD:
        PARAMS: [
          ["eval_mlp", {"in_channels": 64, "dims": [9216, 1000]}],
          ["eval_mlp", {"in_channels": 256, "dims": [9216, 1000]}],
          ["eval_mlp", {"in_channels": 512, "dims": [8192, 1000]}],
          ["eval_mlp", {"in_channels": 1024, "dims": [9216, 1000]}],
          ["eval_mlp", {"in_channels": 2048, "dims": [8192, 1000]}],
          ["eval_mlp", {"in_channels": 2048, "dims": [2048, 1000]}],
        ]


Eval Config settings using SVM training
------------------------------------------

For SVM trainings, we only care about extracting the features from the model. We dump the features on disk and train SVMs. To extract the
features:

Features from several layers of the trunk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, for a ResNet-50 model, to train features from many layers of the model, the example config:

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_ONLY: True
        EXTRACT_TRUNK_FEATURES_ONLY: True   # only extract the features and we will train SVM on these
        LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
          ["res4", ["AvgPool2d", [[8, 8], 3, 0]]],
          ["res5", ["AvgPool2d", [[6, 6], 1, 0]]],
          ["res5avg", ["Identity", []]],
        ]
    TRUNK:
        NAME: resnet
        TRUNK_PARAMS:
          RESNETS:
            DEPTH: 50


Features from the trunk output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, for a ResNet-50 model, to train features from model trunk output, the example config:

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_ONLY: True
        EXTRACT_TRUNK_FEATURES_ONLY: True
      TRUNK:
        NAME: resnet
        TRUNK_PARAMS:
          RESNETS:
            DEPTH: 50


Below, we provide instruction on how to run each benchmark.

Benchmark: ImageNet-1k
---------------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/linear_image_classification/imagenet1k>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/inear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: Places205
---------------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/linear_image_classification/places205>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/inear_image_classification/places205/eval_resnet_8gpu_transfer_places205_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: iNaturalist2018
---------------------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/linear_image_classification/inaturalist18>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/inear_image_classification/inaturalist18/eval_resnet_8gpu_transfer_inaturalist18_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: Linear SVM on VOC07
---------------------------------

VISSL provides :code:`train_svm.py` tool that will first extract features and then train/test SVMs on these features.
The configuration setting for this benchmark is provided `here <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/linear_image_classification/voc07>`_ .

.. code-block:: bash

    python tools/train_svm.py \
      config=benchmark/inear_image_classification/voc07/eval_resnet_8gpu_transfer_voc07_svm \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


.. note::

    Please see VISSL documentation on how to run a given training on **1-gpu, multi-gpu or multi-machine**.

.. note::

    Please see VISSL documentation on how to use the **builtin datasets**.

.. note::

    Please see VISSL documentation on how to use YAML comfiguration system in VISSL to **override specific components like model** of a config file. For example,
    in the above file, user can replace ResNet-50 model with a different architecture like RegNetY-256 etc. easily.
