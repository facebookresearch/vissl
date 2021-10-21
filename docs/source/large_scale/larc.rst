LARC for Large batch size training
====================================

What is LARC
--------------
LARC (Large Batch Training of Convolutional Networks) is a technique proposed by **Yang You, Igor Gitman, Boris Ginsburg** in https://arxiv.org/abs/1708.03888 for improving the convergence of large batch size trainings.
LARC uses the ratio between gradient and parameter magnitudes to calculate an adaptive local learning rate for each individual parameter.

See the `LARC paper <https://arxiv.org/abs/1708.03888>`_ for the calculation of the learning rate. In practice, it modifies the gradients of parameters as a proxy
for modifying the learning rate of the parameters.

How to enable LARC
--------------------

VISSL supports the LARC implementation from `NVIDIA's Apex LARC <https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py>`_. To use LARC, users need to set config option
:code:`OPTIMIZER.use_larc=True`. VISSL exposes LARC parameters that users can tune. Full list of LARC parameters exposed by VISSL:

.. code-block:: yaml

    OPTIMIZER:
      name: "sgd"
      use_larc: False  # supported for SGD only for now
      larc_config:
        clip: False
        eps: 1e-08
        trust_coefficient: 0.001

.. note::

    LARC is currently supported for SGD optimizer only.



Using Apex
~~~~~~~~~~~~~~~

VISSL provides :code:`anaconda` and :code:`pip` packages of Apex (compiled with Optimzed C++ extensions/CUDA kernels). The Apex
packages are provided for all versions of :code:`CUDA (9.2, 10.0, 10.1, 10.2, 11.0), PyTorch >= 1.4 and Python >=3.6 and <=3.9`.

Follow VISSL's instructions to `install apex in pip <https://github.com/facebookresearch/vissl/blob/main/INSTALL.md#step-2-install-pytorch-opencv-and-apex-pip>`_ and instructions to `install apex in conda <https://github.com/facebookresearch/vissl/blob/main/INSTALL.md#step-3-install-apex-conda>`_.
