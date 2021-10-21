Compatibility with Other Libraries
========================================

- VISSL provides several helpful scripts to convert VISSL models to models that are compatible with other libraries like `Detectron2 <https://github.com/facebookresearch/detectron2>`_ and `ClassyVision <https://github.com/facebookresearch/ClassyVision>`_.
- VISSL also provides scripts to convert models from other sources like `Caffe2 models <https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md>`_ in the `paper <https://arxiv.org/abs/1905.01235>`_ to VISSL compatible models.
- `TorchVision <https://github.com/pytorch/vision/tree/main/torchvision/models>`_ models trunks are directly compatible with VISSL and don't require any conversion.

Converting Models VISSL -> {Detectron2, ClassyVision, TorchVision}
-------------------------------------------------------------------
We provide scripts to convert VISSL models to `Detectron2 <https://github.com/facebookresearch/detectron2>`_ and `ClassyVision <https://github.com/facebookresearch/ClassyVision>`_ compatible models.

Converting to Detectron2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All the ResNe(X)t models in VISSL can be converted to Detectron2 weights using the following command:

.. code-block:: bash

    python extra_scripts/convert_vissl_to_detectron2.py \
        --input_model_file <input_model>.pth  \
        --output_model <d2_model>.torch \
        --weights_type torch \
        --state_dict_key_name classy_state_dict


Converting to ClassyVision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All the ResNe(X)t models in VISSL can be converted to Classy Vision weights using the following command:

.. code-block:: bash

    python extra_scripts/convert_vissl_to_classy_vision.py \
        --input_model_file <input_model>.pth  \
        --output_model <d2_model>.torch \
        --state_dict_key_name classy_state_dict


Converting to TorchVision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All the ResNe(X)t models in VISSL can be converted to Torchvision weights using the following command:

.. code-block:: bash

    python extra_scripts/convert_vissl_to_torchvision.py \
        --model_url_or_file <input_model>.pth  \
        --output_dir /path/to/output/dir/ \
        --output_name <my_converted_model>.torch


Converting Caffe2 models -> VISSL
----------------------------------------
We provide conversion of all the `Caffe2 models <https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md>`_ in the `paper <https://arxiv.org/abs/1905.01235>`_.

ResNet-50 models to VISSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Jigsaw model:**

.. code-block:: bash

    python extra_scripts/convert_caffe2_to_torchvision_resnet.py \
        --c2_model <model>.pkl \
        --output_model <pth_model>.torch \
        --jigsaw True --bgr2rgb True


- **Colorization model:**

.. code-block:: bash

    python extra_scripts/convert_caffe2_to_torchvision_resnet.py \
        --c2_model <model>.pkl \
        --output_model <pth_model>.torch \
        --bgr2rgb False


- **Supervised model:**

.. code-block:: bash

    python extra_scripts/convert_caffe2_to_pytorch_rn50.py \
        --c2_model <model>.pkl \
        --output_model <pth_model>.torch \
        --bgr2rgb True


AlexNet models to VISSL
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **AlexNet Jigsaw models:**

.. code-block:: bash

    python extra_scripts/convert_caffe2_to_vissl_alexnet.py \
        --weights_type caffe2 \
        --model_name jigsaw \
        --bgr2rgb True \
        --input_model_weights <model.pkl> \
        --output_model <pth_model>.torch


- **AlexNet Colorization models:**

.. code-block:: bash

    python extra_scripts/convert_caffe2_to_vissl_alexnet.py \
        --weights_type caffe2 \
        --model_name colorization \
        --input_model_weights <model.pkl> \
        --output_model <pth_model>.torch


- **AlexNet Supervised models:**

.. code-block:: bash

    python extra_scripts/convert_caffe2_to_vissl_alexnet.py \
        --weights_type caffe2 \
        --model_name supervised \
        --bgr2rgb True \
        --input_model_weights <model.pkl> \
        --output_model <pth_model>.torch


Converting Models ClassyVision -> VISSL
-------------------------------------------
We provide scripts to convert `ClassyVision <https://github.com/facebookresearch/ClassyVision>`_ models to `VISSL <https://github.com/facebookresearch/vissl>`_ compatible models.

.. code-block:: bash

    python extra_scripts/convert_classy_vision_to_vissl_resnet.py \
        --input_model_file <input_model>.pth  \
        --output_model <d2_model>.torch \
        --depth 50


Converting Official RotNet and DeepCluster models -> VISSL
------------------------------------------------------------

- **AlexNet RotNet model:**

.. code-block:: bash

    python extra_scripts/convert_caffe2_to_vissl_alexnet.py \
        --weights_type torch \
        --model_name rotnet \
        --input_model_weights <model> \
        --output_model <pth_model>.torch


- **AlexNet DeepCluster model:**

.. code-block:: bash

    python extra_scripts/convert_alexnet_models.py \
        --weights_type torch \
        --model_name deepcluster \
        --input_model_weights <model> \
        --output_model <pth_model>.torch
