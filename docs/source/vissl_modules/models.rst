Building Models
===============================

The default model used in vissl is :code:`BaseSSLMultiInputOutputModel`. This model is split into :code:`trunk` that computes features and :code:`head` that computes outputs (projections, classifications etc).

VISSL supports several types of Heads and several types of trunks. VISSL implements a default model :code:`BaseSSLMultiInputOutputModel` which supports the following use cases:

- Model producing single output as in standard supervised ImageNet training.

- Model producing multiple outputs (Multi-task).

- Model producing multiple outputs from different features (layers) from the trunk (useful in linear evaluation of features from several model layers).

- Model that accepts multiple inputs (e.g. image and patches as in PIRL appraoch).

- Model where the trunk is frozen and head is trained.

- Model that supports multiple resolutions inputs as in SwAV.

- Model that is completely frozen and features are extracted.

Users can also implement their own model and specify the model name in :code:`MODEL.NAME`.

Trunks
-------------

VISSL supports many trunks including AlexNet (variants for approaches like Jigsaw, Colorization, RotNet, DeepCluster etc), ResNets, ResNeXt, RegNets, EfficientNet.

To set the trunk, user needs to specify the trunk name in :code:`MODEL.TRUNK.NAME`.

Examples of trunks:

- **Using ResNe(X)ts trunk:**

.. code-block:: yaml

    MODEL:
      TRUNK:
        NAME: resnet
        RESNETS:
          DEPTH: 50
          WIDTH_MULTIPLIER: 1
          NORM: BatchNorm    # BatchNorm | LayerNorm | GroupNorm
          # If using GroupNorm, this sets number of groups. Recommend 32 as a
          # naive suggestion. GroupNorm only available for ResNe(X)t.
          GROUPNORM_GROUPS: 32
          # Use weight-standardized convolutions
          STANDARDIZE_CONVOLUTIONS: False
          GROUPS: 1
          ZERO_INIT_RESIDUAL: False
          WIDTH_PER_GROUP: 64
          # Colorization model uses stride=1 for last layer to retain higher spatial resolution
          # for the pixel-wise task. Torchvision default is stride=2 and all other models
          # use this so we set the default as 2.
          LAYER4_STRIDE: 2

- **Using RegNets trunk**: We follow `RegNets defined in ClassyVision directly <https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/regnet.py>`_ and users can either use a pre-defined ClassyVision RegNet config or define their own.

  - for example, to create a new RegNet config for RegNet-256Gf model (1.3B params):

    .. code-block:: yaml

        MODEL:
          TRUNK:
            NAME: regnet
            REGNET:
              depth: 27
              w_0: 640
              w_a: 230.83
              w_m: 2.53
              group_width: 373

  - To use a pre-defined RegNet config in classy vision example: RegNetY-16gf

    .. code-block:: yaml

        MODEL:
          TRUNK:
            NAME: regnet_y_16gf


Heads
------------

This function creates the heads needed by the module. The head is specified by setting :code:`MODEL.HEAD.PARAMS` in the configuration file.

The :code:`MODEL.HEAD.PARAMS` is a list of Pairs containing parameters for (multiple) heads.

- Pair[0] = Name of Head.
- Pair[1] = kwargs passed to head constructor.

Example of ["name", kwargs] :code:`MODEL.HEAD.PARAMS=["mlp", {"dims": [2048, 128]}]`

Types of Heads one can specify
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Case1: Simple Head containing single module - Single Input, Single output**

.. code-block:: yaml

    MODEL:
      HEAD:
        PARAMS: [
            ["mlp", {"dims": [2048, 128]}]
        ]

- **Case2: Complex Head containing chain of head modules - Single Input, Single output**

.. code-block:: yaml

    MODEL:
      HEAD:
        PARAMS: [
            ["mlp", {"dims": [2048, 1000], "use_bn": False, "use_relu": False}],
            ["siamese_concat_view", {"num_towers": 9}],
            ["mlp", {"dims": [9000, 128]}]
        ]

- **Case3: Multiple Heads (example 2 heads) - Single input, multiple output**: can be used for multi-task learning

.. code-block:: yaml

    MODEL:
      HEAD:
        PARAMS: [
            # head 0
            [
                ["mlp", {"dims": [2048, 128]}]
            ],
            # head 1
            [
                ["mlp", {"dims": [2048, 1000], "use_bn": False, "use_relu": False}],
                ["siamese_concat_view", {"num_towers": 9}],
                ["mlp", {"dims": [9000, 128]}],
            ]
        ]

- **Case4: Multiple Heads (example 5 simple heads) - Single input, multiple output:**: For example, in linear evaluation of models. This attaches a head to each of the layers specified in :code:`MODEL.FEATURE_EVAL_SETTINGS`.

.. code-block:: yaml

    MODEL:
      HEAD:
        PARAMS: [
            ["eval_mlp", {"in_channels": 64, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 256, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 512, "dims": [8192, 1000]}],
            ["eval_mlp", {"in_channels": 1024, "dims": [9216, 1000]}],
            ["eval_mlp", {"in_channels": 2048, "dims": [8192, 1000]}],
        ]

Applying heads on multiple trunk features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the head operates on the trunk output (single or multiple output). However, one can explicitly specify the :code:`input` to heads mapping in the list :code:`MODEL.MULTI_INPUT_HEAD_MAPPING`. This is used in PIRL training.

Assumptions:

- This assumes that the same trunk is used to extract features for the different types of inputs.

- One head only operates on one kind of input, Every individual head can contain several layers as in Case2 above.

:code:`MODEL.MULTI_INPUT_HEAD_MAPPING` specifies Input -> Trunk Features mapping. Like in the single input case, the heads can operate on features from different layers. In this case, we specify :code:`MODEL.MULTI_INPUT_HEAD_MAPPING` to be a list like:

.. code-block:: yaml

    MODEL:
      MULTI_INPUT_HEAD_MAPPING: [
            ["input_key", [list of features heads is applied on]]
      ]

For example: for a model that applies two heads on images and one head on patches:

.. code-block:: yaml

    MODEL:
        MULTI_INPUT_HEAD_MAPPING: [
            ["images", ["res5", "res4"]],
            ["patches", ["res3"]
        ],
