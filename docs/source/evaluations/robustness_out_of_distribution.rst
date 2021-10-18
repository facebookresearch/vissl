Benchmark: Robustness Out-Of-Distribution (OOD)
===========================================================

VISSL provides a standardized benchmark suite to evaluate the feature representation quality of self-supervised pretrained models.

One particularly important type of benchmark is robustness to distribution shift.
These benchmarks are also interesting on models pre-trained using Self-Supervised Learning and have been used in the past in papers such as `CLIP <https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf>`_:
they give a measure of how well an SSL algorithm is able to produce representations free of spurious correlations.

VISSL provides benchmarks for Out-Of-Distribution (OOD) generalisation based on the following datasets `Imagenet-A <https://github.com/hendrycks/natural-adv-examples>`_ , `Imagenet-R <https://github.com/hendrycks/imagenet-r>`_, `Objectnet <https://objectnet.dev/>`_, `Imagenet Real <https://arxiv.org/pdf/2006.07159.pdf>`_, `Imagenet Sketch <https://github.com/HaohanWang/ImageNet-Sketch>`_, and `Imagenet V2 <https://github.com/modestyachts/ImageNetV2>`_.
In those benchmarks, a pre-trained model is either fine-tuned or trained with linear evaluation on Imagenet, and then tested against each dataset.


Benchmark: ImageNet-A
---------------------------

`Imagenet-A <https://github.com/hendrycks/natural-adv-examples>`_ contains a set of 7500 adversarial examples, on 200 of the 1000 classes of Imagenet.
It features images that are typically misclassified by most of the convolutional network architectures when trained on Imagenet.

To run the benchmark with linear evaluation, you can use the following command:

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/robustness_out_of_distribution/imagenet_a/eval_resnet_8gpu_robustness_in1k_linear
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

To run the benchmark with fine-tuning, you can use the following command:

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/robustness_out_of_distribution/imagenet_a/eval_resnet_8gpu_robustness_in1k_fulltune
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>


Benchmark: ImageNet-R
---------------------------

`Imagenet-R <https://github.com/hendrycks/imagenet-r>`_ contains a set of 30000 samples, on 200 of the 1000 classes of Imagenet.
It features images with different renditions that the one features in Imagenet, for instances paintings and sketches, and measures the ability of the model to generalize out-of-distribution.

To run the benchmark with linear evaluation, you can use the following command:

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/robustness_out_of_distribution/imagenet_r/eval_resnet_8gpu_robustness_in1k_linear
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>

To run the benchmark with fine-tuning, you can use the following command:

.. code-block:: bash

    python tools/run_distributed_engines.py \
      config=benchmark/robustness_out_of_distribution/imagenet_r/eval_resnet_8gpu_robustness_in1k_fulltune
      config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<my_weights.torch>
