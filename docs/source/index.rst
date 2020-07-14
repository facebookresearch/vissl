.. VISSL documentation master file, created by
   sphinx-quickstart on Mon Jul 13 07:30:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/facebookresearch/vissl/


VISSL documentation
===================

VISSL is a computer vision library for state-of-the-art Self-Supervised Learning research with
`PyTorch <https://pytorch.org>`_. VISSL aims to accelerate research cycle in self-supervised learning:
from designing a new self-supervised task to evaluating the learned representations.

.. toctree::
   :maxdepth: 1
   :caption: Index

   getting_started


.. toctree::
   :maxdepth: 1
   :caption: Using VISSL Modules

   vissl_modules/train
   vissl_modules/models
   vissl_modules/optimizer
   vissl_modules/criterions
   vissl_modules/meters
   vissl_modules/hooks
   vissl_modules/data


.. toctree::
   :maxdepth: 1
   :caption: Extending VISSL Modules

   extend_modules/train_step
   extend_modules/hooks
   extend_modules/optimizer
   extend_modules/criterions
   extend_modules/meters
   extend_modules/models
   extend_modules/custom_datasets
   extend_modules/data_source
   extend_modules/dataloader
   extend_modules/data_transforms
   extend_modules/data_collators

.. toctree::
   :maxdepth: 1
   :caption: Flowcharts for VISSL execution

   flowcharts/train_workflow
   flowcharts/extract_workflow
   flowcharts/svm_workflow
   flowcharts/nearest_neighbor

.. toctree::
   :maxdepth: 1
   :caption: Training resource setup

   train_resource_setup

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation


.. toctree::
   :maxdepth: 1
   :caption: VISSL Configuration files

   hydra_config


.. toctree::
   :maxdepth: 1
   :caption: Self-supervision approaches

   ssl_approaches/rotnet
   ssl_approaches/jigsaw
   ssl_approaches/npid
   ssl_approaches/clusterfit
   ssl_approaches/pirl
   ssl_approaches/simclr
   ssl_approaches/moco
   ssl_approaches/swav


.. toctree::
   :maxdepth: 1
   :caption: Evaluation tasks

   evaluations/linear_benchmark
   evaluations/full_finetune_in1k
   evaluations/nearest_neighbor
   evaluations/semi_supervised
   evaluations/object_detection


.. toctree::
   :maxdepth: 1
   :caption: Visualization with Tensorboard

   visualization

.. toctree::
   :maxdepth: 1
   :caption: Large Scale Self-Supervised learning

   large_scale/larc
   large_scale/queue_dataset
   large_scale/stateful_sampler
   large_scale/mixed_precision
   large_scale/distributed_training

.. toctree::
   :maxdepth: 1
   :caption: Compatibility with other Libraries

   compatibility_libraries

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   contributing


.. toctree::
   :maxdepth: 1
   :caption: Contact

   contacts


.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/index
