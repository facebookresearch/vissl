Benchmark: Linear Image Classification
===========================================================

VISSL provides a standardized benchmark suite to evaluate the feature representation quality of self-supervised pretrained models. A popular
evaluation protocol is to freeze the model traink and train linear classifiers on several layers of the model on some target datasets (like ImageNet-1k, Places205, VOC07, iNaturalist2018).
In VISSL, we support linear evaluations on all the common datasets. We also provide standard set of hyperparams for various approaches
in order to reproduce the model performance in SSL literature. For reproducibility, see `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_.

.. note::

    To run the benchmark, we recommend using the standard set of hyperparams provided by VISSL as these hyperparams reproduce results of a large number of self-supervised approaches.
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
        RESNETS:
            DEPTH: 50
      HEAD:
        PARAMS: [
            ["mlp", {"dims": [2048, 1000]}],
        ]


Attaching MLP head to trunk output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To attach a linear classifier to multiple layers of the model following Zhang et. al style which has :code:`BN -> FC` as the head, use :code:`eval_mlp` head. For example, for a ResNet-50 model:

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
        RESNETS:
          DEPTH: 50


Below, we provide instruction on how to run each benchmark.

Benchmark: ImageNet-1k
---------------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/imagenet1k>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: Places205
---------------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/places205>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/places205/eval_resnet_8gpu_transfer_places205_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: iNaturalist2018
---------------------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/inaturalist18>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/inaturalist18/eval_resnet_8gpu_transfer_inaturalist18_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for iNaturalist2018 is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.

Benchmark: CIFAR-10
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/cifar10>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/cifar10/eval_resnet_8gpu_transfer_cifar10_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: CIFAR-100
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/cifar100>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/cifar100/eval_resnet_8gpu_transfer_cifar100_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: MNIST
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/mnist>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/mnist/eval_resnet_8gpu_transfer_mnist_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: STL-10
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/stl10>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/stl10/eval_resnet_8gpu_transfer_stl10_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: SVHN
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/svhn>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/svhn/eval_resnet_8gpu_transfer_svhn_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: Caltech-101
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/caltech101>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/caltech101/eval_resnet_8gpu_transfer_caltech101_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for Caltech-101 is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: Describable Textures
--------------------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/dtd>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/dtd/eval_resnet_8gpu_transfer_dtd_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for DTD is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: FGVC Aircrafts
---------------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/aircrafts>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/aircrafts/eval_resnet_8gpu_transfer_aircrafts_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for FGVC Aircrafts is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: FOOD-101
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/food101>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/food101/eval_resnet_8gpu_transfer_food101_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for FOOD-101 is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: GTSRB
-----------------------

The configuration setting for the German Traffic Sign Recognition Benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/gtsrb>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/gtsrb/eval_resnet_8gpu_transfer_gtsrb_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for GTSRB is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: Oxford Flowers
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/oxford_flowers>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/oxford_flowers/eval_resnet_8gpu_transfer_oxford_flowers_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for Oxford Flowers is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: Oxford Pets
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/oxford_pets>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/oxford_pets/eval_resnet_8gpu_transfer_oxford_pets_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for Oxford Pets is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: SUN397
-----------------------

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/sun397>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/sun397/eval_resnet_8gpu_transfer_sun397_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for SUN397 is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: UCF-101
-----------------------

The UCF-101 benchmark evaluates the classification performance on human actions from a single image (the middle frame of the UCF101 dataset).

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/ucf101>`_.

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/ucf101/eval_resnet_8gpu_transfer_ucf101_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for UCF-101 is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_. This script will handle the transformation from videos to images by extracting the middle frame of each of the videos.


Benchmark: EuroSAT
----------------------------

The EuroSAT benchmark evaluates the classification performance on a specialized task (satellite imaging).

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/euro_sat>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/euro_sat/eval_resnet_8gpu_transfer_euro_sat_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for EuroSAT is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: Patch Camelyon
----------------------------

The Patch Camelyon (PCAM) benchmark evaluates the classification performance on a specialized task (medical task).

The configuration setting for this benchmark
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/pcam>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/pcam/eval_resnet_8gpu_transfer_pcam_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

A script to automatically prepare the data for Patch Camelyon is available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: CLEVR
-------------------

The CLEVR benchmarks evaluate the understanding of the structure of a 3D scene by:

- CLEVR/Count: counting then number of objects in the scene
- CLEVR/Dist: estimating the distance to the closest object in the scene

The configuration setting for these benchmarks
is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/clever_count>`_ and `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/clevr_dist>`_.

.. code-block:: bash

    # For CLEVR Count
    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/clevr_count/eval_resnet_8gpu_transfer_clevr_count_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

    # For CLEVR Dist
    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/clevr_dist/eval_resnet_8gpu_transfer_clevr_dist_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

Scripts to automatically prepare the data for the CLEVR benchmarks are available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: dSprites
----------------------

The dSprites benchmarks evaluate the understanding of the positional information in a synthetic 2D scene by:

- dSprites/location: estimating the X position of a sprite
- dSprites/orientation: estimating the orientation of a sprite

The configuration setting for these benchmarks
is provided under the `dsprites <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/dsprites>`_ folder.

.. code-block:: bash

    # For dSprites location
    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/dsprites//eval_resnet_8gpu_transfer_dsprites_loc_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

    # For dSprites orientation
    python tools/run_distributed_engines.py \
      config=benchmark/linear_image_classification/dsprites/eval_resnet_8gpu_transfer_dsprites_orient_linear \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

Scripts to automatically prepare the data for the dSprites benchmarks are available `here <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_.


Benchmark: Linear SVM on VOC07
---------------------------------

VISSL provides :code:`train_svm.py` tool that will first extract features and then train/test SVMs on these features.
The configuration setting for this benchmark is provided `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification/voc07>`_ .

.. code-block:: bash

    python tools/train_svm.py \
      config=benchmark/linear_image_classification/voc07/eval_resnet_8gpu_transfer_voc07_svm \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


.. note::

    Please see VISSL documentation on how to run a given training on **1-gpu, multi-gpu or multi-machine**.

.. note::

    Please see VISSL documentation on how to use the **builtin datasets**.

.. note::

    Please see VISSL documentation on how to use YAML comfiguration system in VISSL to **override specific components like model** of a config file. For example,
    in the above file, user can replace ResNet-50 model with a different architecture like RegNetY-256 etc. easily.
