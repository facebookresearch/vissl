Mixed precision training (fp16)
===================================

Many self-supervised approaches leverage mixed precision training by default for better training speed and reducing the model memory requirement.
For this, you can either use `NVIDIA Apex Library with AMP <https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use>`_ or `Pytorch's implementation <https://pytorch.org/docs/stable/notes/amp_examples.html>`_.

If using Apex, users can tune the AMP level to the levels supported by NVIDIA. See `this for details on Apex amp levels <https://nvidia.github.io/apex/amp.html#opt-levels>`_.

To use Mixed precision training, one needs to set the following parameters in configuration file:

.. code-block:: yaml

    MODEL:
      AMP_PARAMS:
        USE_AMP: False
        # Only applicable for Apex AMP_TYPE
        # Use O1 as it is robust and stable than O3. If you want to use O3, we recommend the following setting:
        # {"opt_level": "O3", "keep_batchnorm_fp32": True, "master_weights": True, "loss_scale": "dynamic"}
        AMP_ARGS: {"opt_level": "O1"}
        # we support pytorch amp as well which is availale in pytorch>=1.6.
        AMP_TYPE: "apex"  # apex | pytorch


Using Apex
~~~~~~~~~~~~~~~

In order to use Apex, VISSL provides :code:`anaconda` and :code:`pip` packages of Apex (compiled with Optimzed C++ extensions/CUDA kernels). The Apex
packages are provided for all versions of :code:`CUDA (9.2, 10.0, 10.1, 10.2, 11.0), PyTorch >= 1.4 and Python >=3.6 and <=3.9`.

Follow VISSL's instructions to `install apex in pip <https://github.com/facebookresearch/vissl/blob/main/INSTALL.md#step-2-install-pytorch-opencv-and-apex-pip>`_ and instructions to `install apex in conda <https://github.com/facebookresearch/vissl/blob/main/INSTALL.md#step-3-install-apex-conda>`_.
