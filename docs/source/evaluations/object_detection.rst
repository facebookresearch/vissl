Benchmark task: Object Detection
===========================================================

Object Detection is a very common benchmark for evaluating feature representation quality. In VISSL, we use `Detectron2 <https://github.com/facebookresearch/detectron2>`_ for the object detection benchmark.

This benchmark involves 2 steps:

- **Step1**: Converting the self-supervised model weights so they are compatible with :code:`Detectron2`.

- **Step2**: Using the converted weights in Step1, run the benchmark.


Converting weights to Detectron2
----------------------------------

VISSL provides a `script to convert the weight of VISSL compatible models to Detectron2 <https://github.com/facebookresearch/vissl/blob/main/extra_scripts/convert_vissl_to_detectron2.py>`_.
We recommend users to adapt this script to suit their needs (different model architecture etc).

To run the script, follow the command:

.. code-block:: bash

    python extra_scripts/convert_vissl_to_detectron2.py \
      --input_model_file <input_model_path>.torch  \
      --output_model <converted_d2_model_path>.torch \
      --weights_type torch \
      --state_dict_key_name classy_state_dict

The script above converts ResNe(X)ts models in VISSL to the models compatible with ResNe(X)ts in Detectron2.


Benchmark: Faster R-CNN on VOC07
-----------------------------------------

VISSL provides the YAML configuration files for :code:`Detectron2` for the benchmark task of Object detection using :code:`Faster R-CNN` on VOC07.
The configuration files are available `here <https://github.com/facebookresearch/vissl/blob/main/configs/config/benchmark/object_detection/voc07/rn50_transfer_voc07_detectron2_e2e.yaml>`_.
To run the benchmark, VISSL provides a python script that closely follows `MoCo object detection <https://github.com/facebookresearch/moco/blob/main/detection/train_net.py>`_.

Please make sure to install Detectron2 following the `Detectron2 Installation instructions <https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md>`_.

To run the benchmark:

.. code-block:: bash

    python tools/object_detection_benchmark.py \
        --config-file ../configs/config/benchmark/object_detection/voc07/rn50_transfer_voc07_detectron2_e2e.yaml \
        --num-gpus 8 MODEL.WEIGHTS <converted_d2_model_path>.torch

.. note::

    We recommend users to consult Detectron2 documentation for how to use the configuration files and how to run the trainings.


PIRL object detection
~~~~~~~~~~~~~~~~~~~~~~

To reproduce the object detection benchmark, the LR and warmup iterations are different. Use the following command:

.. code-block:: bash

    python tools/object_detection_benchmark.py \
        --config-file ../configs/config/benchmark/object_detection/voc07/pirl_npid/rn50_transfer_voc07_pirl_npid.yaml \
        --num-gpus 8 MODEL.WEIGHTS <converted_d2_model_path>.torch

Benchmark: Faster R-CNN on VOC07+12
--------------------------------------------

VISSL provides the YAML configuration files for :code:`Detectron2` for the benchmark task of Object detection using :code:`Faster R-CNN` on VOC07+12.
The configuration files are available `here <https://github.com/facebookresearch/vissl/blob/main/configs/config/benchmark/object_detection/voc0712>`_.
To run the benchmark, VISSL provides a python script that closely follows `MoCo object detection <https://github.com/facebookresearch/moco/blob/main/detection/train_net.py>`_.

Please make sure to install Detectron2 following the `Detectron2 Installation instructions <https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md>`_.

For the VOC07+12 benchmark, most self-supervised approaches use their set of hyperparams. VISSL provides the settings used in

`Scaling and Benchmarking Self-Supervised Visual Representation Learning <https://arxiv.org/pdf/1905.01235.pdf>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python tools/object_detection_benchmark.py \
        --config-file ../configs/config/benchmark/object_detection/voc0712/iccv19/rn50_transfer_voc0712_detectron2_e2e.yaml \
        --num-gpus 8 MODEL.WEIGHTS <converted_d2_model_path>.torch

`MoCoV2 <https://arxiv.org/abs/2003.04297>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python tools/object_detection_benchmark.py \
        --config-file ../configs/config/benchmark/object_detection/voc0712/mocoV2/rn50_transfer_voc0712_detectron2_e2e.yaml \
        --num-gpus 8 MODEL.WEIGHTS <converted_d2_model_path>.torch

`PIRL <https://arxiv.org/abs/1912.01991>`_ and `NPID <https://arxiv.org/abs/1805.01978>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: bash

    python tools/object_detection_benchmark.py \
        --config-file ../configs/config/benchmark/object_detection/voc0712/pirl_npid/rn50_transfer_voc0712_npid_pirl.yaml \
        --num-gpus 8 MODEL.WEIGHTS <converted_d2_model_path>.torch

`SimCLR <https://arxiv.org/abs/2002.05709>`_ and `SwAV <https://arxiv.org/abs/2006.09882>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python tools/object_detection_benchmark.py \
        --config-file ../configs/config/benchmark/object_detection/voc0712/simclr_swav/rn50_transfer_voc0712_simclr_swav.yaml \
        --num-gpus 8 MODEL.WEIGHTS <converted_d2_model_path>.torch


Benchmark: Mask R-CNN on COCO
-----------------------------

Benchmarking on COCO introduces many variants (model architecture, FPN or not, C4). We provide config files for all the variants `here <https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/object_detection/COCOInstance>`_
and encourage users to pick the settings most suitable for their needs.

Benchmarking on COCO is not as widely adopted (compared to VOC07 and voc0712 evaluation) in self-supervision literature. This benchmark has been demonstrated extensively in `MoCoV2 <https://arxiv.org/abs/1911.05722>`_ paper and we encourage users to refer to the paper.

An example run:

.. code-block:: bash

    python tools/object_detection_benchmark.py \
        --config-file ../configs/config/benchmark/object_detection/COCOInstance/sbnExtraNorm_precBN_r50_c4_coco.yaml \
        --num-gpus 8 MODEL.WEIGHTS <converted_d2_model_path>.torch
