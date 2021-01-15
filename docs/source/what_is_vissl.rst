What is VISSL?
==================

VISSL is a computer VIsion library for state-of-the-art Self-Supervised Learning, based on `PyTorch <https://github.com/pytorch/pytorch>`_ and `Classy Vision <https://github.com/facebookresearch/ClassyVision>`_, developed by `FAIR (Facebook AI Research) <https://ai.facebook.com/>`_. A few key features of VISSL are:

- :code:`Ease of Usability`: easy to use using yaml configuration system based on `Hydra <https://github.com/facebookresearch/hydra>`_.
- :code:`Modular`: Easy to design new tasks and reuse the existing components from other tasks (objective functions, model heads, data preparation). The modular components are simple drop-in replacements in yaml config files.
- :code:`Scalability`: Easy to train model on 1-gpu, multi-gpu and multi-node. Several components for large scale trainings provided as simple config file plugs: Activation checkpointing, ZeRO, FP16, LARC, Stateful data sampler, data class to handle invalid images, large model backbones like RegNets, ResNeXt etc.
- :code:`Reproducible implementation of SOTA in SSL`: All existing SOTA in SSL are implemented: **SwAV, SimCLR, MoCo, PIRL, NPID, NPID++, DeepCluster v2, ClusterFit, RotNet, Jigsaw**. Also supports supervised trainings.
- :code:`Benchmark suite`: Variety of benchmarks tasks including linear image classification (places205, imagenet1k, voc07), full finetuning, semi-supervised benchmark, nearest neighbor benchmark, instance retrieval test, object detection (Pascal VOC and COCO) etc are
- :code:`Fully PyTorch based`: Completely based on PyTorch
- :code:`Model Zoo`: Over 100 pre-trained self-supervised model weights

We hope that VISSL will democratize self-supervised learning and accelerate advancements in self-supervised learning. We also hope that it will enable research in some important research directions like Generalizability of models etc.

Hope you enjoy using VISSL!
