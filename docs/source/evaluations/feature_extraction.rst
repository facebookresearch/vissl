How to Extract Features
===========================================================

Given a pre-trained models, VISSL makes it easy to extract the features for the model on the datasets. VISSL seamlessly supports TorchVision models and in general, to load non-VISSL models, please
follow our documentation for loading models.

To extract the features for a model that VISSL can load, users need 2 things:

- **config file**: the configuration file should clearly specify what layers of the model should features be extracted from.

- **set the correct engine_name**: in VISSL, we have two types of engine - a) training, b) feature extraction. Users must set :code:`engine_name=extract_features` in the yaml config file.

.. note::

    The SVM training and Nearest Neighbor benchmark workflows don't require setting the :code`engine_name` because the provided
    tools :code:`train_svm` and :code:`nearest_neighbor_test` explicitly add the feature extraction step.


Config File for Feature Extraction
------------------------------------------

Using the following examples, set the config options for your desired use case of feature extraction. Following examples are for ResNet-50 but users can use their model.

Extract features from several layers of the trunk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_ONLY: True
        EXTRACT_TRUNK_FEATURES_ONLY: True
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


Extract features of the trunk output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_ONLY: True
        EXTRACT_TRUNK_FEATURES_ONLY: True
        SHOULD_FLATTEN_FEATS: False
      TRUNK:
        NAME: resnet
        TRUNK_PARAMS:
          RESNETS:
            DEPTH: 50


Extract features of the model head output (self-supervised head)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a given self-supervised approach, to extract the features of the model head (this is very common use case where the model head is a projection head and projects the trunk features into a low-dimensional space),
The config settings should look like below. The example below is for SimCLR head + ResNet-50. Users can replace the :code:`MODEL.HEAD.PARAMS` with the head settings used in the respective
self-supervised model training.

.. code-block:: yaml

    MODEL:
      FEATURE_EVAL_SETTINGS:
        EVAL_MODE_ON: True
        FREEZE_TRUNK_AND_HEAD: True
        EVAL_TRUNK_AND_HEAD: True
      TRUNK:
        NAME: resnet
        TRUNK_PARAMS:
          RESNETS:
            DEPTH: 50
      HEAD:
        PARAMS: [
          ["mlp", {"dims": [2048, 2048], "use_relu": True}],
          ["mlp", {"dims": [2048, 128]}],
        ]
        
.. note::
    
    The config files have option :code:`MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS` which can be set to True to flatten the extracted features to :code:`NxD` dimensions. By default, VISSL doesn't flatten extracted features and return the features as is.

How to extract features
--------------------------

Once users have the desired config file, user can extract features using the following command. VISSL also provides the config files `here <https://github.com/facebookresearch/vissl/tree/master/configs/config/feature_extraction>`_ that users can modify/adapt to their needs.

.. code-block:: bash

    python tools/run_distributed_engines.py \
        config=feature_extraction/extract_resnet_in1k_8gpu \
        +config/feature_extraction/trunk_only=rn50_layers \
        config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>
