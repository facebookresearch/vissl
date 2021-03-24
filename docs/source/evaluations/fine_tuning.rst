Benchmark task: Fine-tuning
===========================================================

Using a self-supervised model to initialize a network and further tune the weights on the target task is a very common evaluation protocol.
This benchmark requires only initializing the model and no other settings in :code:`MODEL.FEATURE_EVAL_SETTINGS` are needed unlike other benchmark tasks.

Fine-tuning on a limit set of data is also an import evaluation protocol, for it reflect one of the common use cases of Self-Supervised Learning algorithms.
A lot of these low shot fine-tuning benchmarks are directly inspired from the `VTAB <https://arxiv.org/pdf/1910.04867.pdf>`_ paper.


Benchmark: ImageNet-1k
------------------------------------------

The configuration for full fine-tuning on Imagenet is available in `benchmark/fulltune/imagenet1k <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/fulltune/imagenet1k>`_ and can be run as follows:

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

The configuration for full fine-tuning on Places205 is available in `benchmark/fulltune/places205 <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/fulltune/places205>`_ and can be run as follows:

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/fulltune/places205/eval_resnet_8gpu_transfer_places205_fulltune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: CIFAR-100
---------------------------

The configuration for low shot (1000 samples) fine-tuning on CIFAR-100 provided `here <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/cifar100>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/cifar100/eval_resnet_8gpu_transfer_cifar100_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

For fine-tuning on the full dataset, just add this option to the command line :code:`config.DATA.TRAIN.DATA_LIMIT=-1`.


Benchmark: CLEVR
--------------------

The CLEVR benchmarks evaluate the understanding of the structure of a 3D scene by:

- CLEVR/Count: counting then number of objects in the scene
- CLEVR/Dist: estimating the distance to the closest object in the scene

The configurations for low shot (1000 samples) fine-tuning on CLEVR are available under the
`clever_count <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/clever_count>`_ and `clevr_dist <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/clevr_dist>`_ folders.

.. code-block:: bash

    # For CLEVR Count
    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/clevr_count/eval_resnet_8gpu_transfer_clevr_count_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

    # For CLEVR Dist
    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/clevr_dist/eval_resnet_8gpu_transfer_clevr_dist_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

For fine-tuning on the full dataset, just add this option to the command line :code:`config.DATA.TRAIN.DATA_LIMIT=-1`.

Scripts to automatically prepare the data for the CLEVR benchmarks are available `here <https://github.com/facebookresearch/vissl/tree/master/extra_scripts>`_.


Benchmark: dSprites
----------------------

The dSprites benchmarks evaluate the understanding of the positional information in a synthetic 2D scene by:

- dSprites/location: estimating the X position of a sprite
- dSprites/orientation: estimating the orientation of a sprite

The configurations for low shot (1000 samples) fine-tuning on dSprites
are provided under the `dsprites_loc <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/dsprites_loc>`_ and `dsprites_orient <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/dsprites_orient>`_ folders.

.. code-block:: bash

    # For dSprites location
    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/dsprites_loc/eval_resnet_8gpu_transfer_dsprites_loc_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

    # For dSprites orientation
    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/dsprites_orient/eval_resnet_8gpu_transfer_dsprites_orient_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

For fine-tuning on the full dataset, just add this option to the command line :code:`config.DATA.TRAIN.DATA_LIMIT=-1`.

Scripts to automatically prepare the data for the dSprites benchmarks are available `here <https://github.com/facebookresearch/vissl/tree/master/extra_scripts>`_.


Benchmark: EuroSAT
----------------------------

The EuroSAT benchmark evaluates the classification performance on a specialized task (satellite imaging).
The configuration for low shot (1000 samples) fine-tuning on EuroSAT
is provided under the `euro_sat <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/euro_sat>`_ folder.

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/euro_sat/eval_resnet_8gpu_transfer_euro_sat_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

For fine-tuning on the full dataset, just add this option to the command line :code:`config.DATA.TRAIN.DATA_LIMIT=-1`.

A script to automatically prepare the data for EuroSAT is available `here <https://github.com/facebookresearch/vissl/tree/master/extra_scripts>`_.


Benchmark: KITTI/Dist
----------------------------

The KITTI/Dist benchmark evaluates the transfer of a pre-trained model to a distance estimation task in a self-driving environment.
The configuration for low shot (1000 samples) fine-tuning on KITTI/Dist
is provided under the `kitti_dist <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/kitti_dist>`_ folder.

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/kitti_dist/eval_resnet_8gpu_transfer_kitti_dist_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

For fine-tuning on the full dataset, just add this option to the command line :code:`config.DATA.TRAIN.DATA_LIMIT=-1`.

A script to automatically prepare the data for KITTI/Dist is available `here <https://github.com/facebookresearch/vissl/tree/master/extra_scripts>`_.


Benchmark: Patch Camelyon
----------------------------

The Patch Camelyon (PCAM) benchmark evaluates the classification performance on a specialized task (medical task).
The configuration for low shot (1000 samples) fine-tuning on PCAM
is provided under the `pcam <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/pcam>`_ folder.

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/pcam/eval_resnet_8gpu_transfer_pcam_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

For fine-tuning on the full dataset, just add this option to the command line :code:`config.DATA.TRAIN.DATA_LIMIT=-1`.

A script to automatically prepare the data for Patch Camelyon is available `here <https://github.com/facebookresearch/vissl/tree/master/extra_scripts>`_.


Benchmark: Small NORB
------------------------

The Small NORB benchmarks evaluate the understanding of the structure of a 3D scene by:

- snorb/azimuth: estimating the azimuth of the object
- snorb/elevation: estimating the elevation of the image

The configurations for low shot (1000 samples) fine-tuning on Small NORB
are provided under the `small_norb_azimuth <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/small_norb_azimuth>`_ and `small_norb_elevation <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/small_norb_elevation>`_ folders.

.. code-block:: bash

    # For dSprites location
    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/small_norb_azimuth/eval_resnet_8gpu_transfer_snorb_azimuth_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

    # For dSprites orientation
    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/small_norb_elevation/eval_resnet_8gpu_transfer_snorb_elevation_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

For fine-tuning on the full dataset, just add this option to the command line :code:`config.DATA.TRAIN.DATA_LIMIT=-1`.

Scripts to automatically prepare the data for the Small NORB benchmarks are available `here <https://github.com/facebookresearch/vissl/tree/master/extra_scripts>`_.


Benchmark: SVHN
-----------------------

The configuration for low shot (1000 samples) fine-tuning on SVHN is provided `here <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/low_shot_fine_tune/svhn>`_ .

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/low_shot_fine_tune/svhn/eval_resnet_8gpu_transfer_svhn_low_tune \
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

For fine-tuning on the full dataset, just add this option to the command line :code:`config.DATA.TRAIN.DATA_LIMIT=-1`.


.. note::

    Please see VISSL documentation on how to run a given training on **1-gpu, multi-gpu or multi-machine**.

.. note::

    Please see VISSL documentation on how to use the **builtin datasets** if you want to run this benchmark on a different target task..

.. note::

    Please see VISSL documentation on how to use YAML comfiguration system in VISSL to **override specific components like model** of a config file. For example,
    in the above file, user can replace ResNet-50 model with a different architecture like RegNetY-256 etc. easily.
