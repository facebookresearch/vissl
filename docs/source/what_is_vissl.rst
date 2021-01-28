What is VISSL?
==================

.. image:: _static/img/vissl-logo.png


VISSL is a computer VIsion library for state-of-the-art Self-Supervised Learning research with `PyTorch <https://pytorch.org>`_. VISSL aims to accelerate research cycle in self-supervised learning: from designing a new self-supervised task to evaluating the learned representations. Key features include:

- :code:`Reproducible implementation of SOTA in Self-Supervision`: All existing SOTA in Self-Supervision are implemented - `SwAV <https://arxiv.org/abs/2006.09882>`_, `SimCLR <https://arxiv.org/abs/2002.05709>`_, `MoCo(v2) <https://arxiv.org/abs/1911.05722>`_, `PIRL <https://arxiv.org/abs/1912.01991>`_, `NPID <https://arxiv.org/abs/1912.01991>`_, `NPID++ <https://arxiv.org/abs/1912.01991>`_, `DeepClusterV2 <https://arxiv.org/abs/2006.09882>`_, `ClusterFit <https://openaccess.thecvf.com/content_CVPR_2020/papers/Yan_ClusterFit_Improving_Generalization_of_Visual_Representations_CVPR_2020_paper.pdf>`_, `RotNet <https://arxiv.org/abs/1803.07728>`_, `Jigsaw <https://arxiv.org/abs/1603.09246>`_. Also supports supervised trainings.

- :code:`Benchmark suite`: Variety of benchmarks tasks including `linear image classification (places205, imagenet1k, voc07, inaturalist) <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/linear_image_classification>`_, `full finetuning <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/imagenet1k_fulltune>`_, `semi-supervised benchmark <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/semi_supervised>`_, `nearest neighbor benchmark <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/nearest_neighbor>`_, `object detection (Pascal VOC and COCO) <https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/object_detection>`_.

- :code:`Ease of Usability`: easy to use using yaml configuration system based on `Hydra <https://github.com/facebookresearch/hydra>`_.

- :code:`Modular`: Easy to design new tasks and reuse the existing components from other tasks (objective functions, model trunk and heads, data transforms, etc.). The modular components are simple *drop-in replacements* in yaml config files.

- :code:`Scalability`: Easy to train model on 1-gpu, multi-gpu and multi-node. Several components for large scale trainings provided as simple config file plugs: `Activation checkpointing <https://pytorch.org/docs/stable/checkpoint.html>`_, `ZeRO <https://arxiv.org/abs/1910.02054>`_, `FP16 <https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use>`_, `LARC <https://arxiv.org/abs/1708.03888>`_, Stateful data sampler, data class to handle invalid images, large model backbones like `RegNets <https://arxiv.org/abs/2003.13678>`_, etc.

- :code:`Model Zoo`: Over *60 pre-trained self-supervised model* weights.

We hope that VISSL will democratize self-supervised learning and accelerate advancements in self-supervised learning. We also hope that it will enable research in some important research directions like Generalizability of models etc.

Hope you enjoy using VISSL!
