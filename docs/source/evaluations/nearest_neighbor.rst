Benchmark: Nearest Neighbor k-means
===========================================================

VISSL supports Nearest Neighbor evaluation task using k-means. We closely follow the benchmark setup from Zhirong Wu et al. `here <https://github.com/zhirongw/lemniscate.pytorch#nearest-neighbor>`_.
For the Nearest neighbor evaluation, the process involves 2 steps:

- **Step1**: Extract the relevant features from the model for both training and validation set.

- **Step2**: Perform k-means clustering and evaluation on these features

VISSL provides a dedicated tool :code:`tools/nearest_neighbor_test.py` that performs both Step-1 and Step-2 above.

.. note::

    To run the benchmark, we recommend using the standard set of hyperparams provided by VISSL as these hyperparams reproduce results of large number of self-supervised approaches. Users are however free to modify the hyperparams to suit their evaluation criterion.


Eval Config settings using MLP head
--------------------------------------

Set the following in the config file to enable the feature evaluation properly.

kNN on many layers of the trunk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the Step1, if we want to extract features from many layers of the trunk, the config setting should look like below. For example for a ResNet-50 model:

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_ONLY: True  # only freeze the trunk
        EXTRACT_TRUNK_FEATURES_ONLY: True   # we extract features from the trunk only
        SHOULD_FLATTEN_FEATS: False   # don't flatten the features and return as is
        LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
          ["res4", ["AvgPool2d", [[8, 8], 3, 0]]],
          ["res5", ["AvgPool2d", [[6, 6], 1, 0]]],
        ]
      TRUNK:
        NAME: resnet
        RESNETS:
          DEPTH: 50

kNN on the trunk output
~~~~~~~~~~~~~~~~~~~~~~~~~

If we want to perform kNN only on the trunk output, the configuration setting should look like below. For example, for a ResNet-50 model:

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_ONLY: True  # only freeze the trunk
        EXTRACT_TRUNK_FEATURES_ONLY: True   # we extract features from the trunk only
        SHOULD_FLATTEN_FEATS: False   # don't flatten the features and return as is
      TRUNK:
        NAME: resnet
        RESNETS:
          DEPTH: 50

kNN on the model head output (self-supervised head)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a given self-supervised approach, we want to perform kNN on the output of the model head. This is very common where the model head is a projection head and projects the trunk features into a low-dimensional space.
The config settings should look like below. The example below is for SimCLR head + ResNet-50. Users can replace the :code:`MODEL.HEAD.PARAMS` with the head settings used in the respective
self-supervised model training.

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_AND_HEAD: True   # both head and trunk will be frozen (including BN in eval mode)
        EVAL_TRUNK_AND_HEAD: True     # initialized the model head as well from weights
      TRUNK:
        NAME: resnet
        RESNETS:
          DEPTH: 50
      HEAD:
        # SimCLR 2-layer model head structure
        PARAMS: [
          ["mlp", {"dims": [2048, 2048], "use_relu": True}],
          ["mlp", {"dims": [2048, 128]}],
        ]


Benchmark: ImageNet-1k
------------------------------

VISSL provides configuration settings for the benchmark `here <https://github.com/facebookresearch/vissl/blob/main/configs/config/benchmark/nearest_neighbor/eval_resnet_8gpu_in1k_kNN.yaml>`_.

To run the benchmark:

.. code-block:: bash

    python tools/nearest_neighbor_test.py config=benchmark/nearest_neighbor/eval_resnet_8gpu_in1k_kNN \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

Benchmark: Places205
----------------------------------

VISSL provides configuration settings for the benchmark `here <https://github.com/facebookresearch/vissl/blob/main/configs/config/benchmark/nearest_neighbor/eval_resnet_8gpu_places205_kNN.yaml>`_.

To run the benchmark:

.. code-block:: bash

    python tools/nearest_neighbor_test.py config=benchmark/nearest_neighbor/eval_resnet_8gpu_places205_kNN \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


.. note::

    Please see VISSL documentation on how to run a given training on **1-gpu, multi-gpu or multi-machine**.

.. note::

    Please see VISSL documentation on how to use the **builtin datasets**.

.. note::

    Please see VISSL documentation on how to use YAML comfiguration system in VISSL to **override specific components like model** of a config file. For example,
    in the above file, user can replace ResNet-50 model with a different architecture like RegNetY-256 etc. easily.
