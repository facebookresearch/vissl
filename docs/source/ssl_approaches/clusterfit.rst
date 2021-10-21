Train ClusterFit model
===============================

VISSL reproduces the self-supervised approach **ClusterFit: Improving Generalization of Visual Representations**
proposed by **Xueting Yan, Ishan Misra, Abhinav Gupta, Deepti Ghadiyaram, Dhruv Mahajan** in `this paper <https://openaccess.thecvf.com/content_CVPR_2020/papers/Yan_ClusterFit_Improving_Generalization_of_Visual_Representations_CVPR_2020_paper.pdf>`_.

How to train ClusterFit model
---------------------------------

VISSL provides a yaml configuration file containing the exact hyperparameter settings to reproduce the model. VISSL implements
all the components including data augmentations, collators etc required for this approach.

ClusterFit approach involves 2 steps:

* **Step1**: Using a pre-trained model (could be trained any way), the features are extracted on the training dataset (like ImageNet). The extracted features are clustered via k-means into N clusters (for example: 16000 clusters). For faster clustering, libraries like `FAISS <https://github.com/facebookresearch/faiss>`_ can be used (supported in VISSL). The cluster centroids are treated as the labels for the images and used for training in the next step.

* **Step2**: The model is trained (scratch initialization) but using the labels generated in Step 1.

To train ResNet-50 model on 8-gpus on ImageNet-1K dataset and using RotNet model to extract features:

.. code-block:: bash

    # Step1: Extract features
    python tools/run_distributed_engines.py config=pretrain/clusterfit/cluster_features_resnet_8gpu_rotation_in1k \
        config.MODEL.WEIGHTS_INIT.PARAMS_FILE=<vissl_compatible_weights.torch>

    # Step2: Train clusterFit model
    python tools/run_distributed_engines.py config=pretrain/clusterfit/clusterfit_resnet_8gpu_imagenet \
        config.DATA.TRAIN.LABEL_PATHS=[<labels_file_from_step1.npy>]


The full set of hyperparams supported by VISSL for ClusterFit Step-1 include:


.. code-block:: yaml

    CLUSTERFIT:
      NUM_CLUSTERS: 16000
      # currently we only support faiss backend for clustering.
      CLUSTER_BACKEND: faiss
      # how many iterations to use for faiss
      N_ITER: 50
      FEATURES:
        DATA_PARTITION: TRAIN
        DATASET_NAME: imagenet1k
        LAYER_NAME: res5


How to use other pre-trained models in VISSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VISSL supports Torchvision model trunks out of the box. Generally, for loading any non-VISSL model, one needs to correctly set the following configuration options:

.. code-block:: yaml

    WEIGHTS_INIT:
      # path to the .torch weights files
      PARAMS_FILE: ""
      # name of the state dict. checkpoint = {"classy_state_dict": {layername:value}}. Options:
      #   1. classy_state_dict - if model is trained and checkpointed with VISSL.
      #      checkpoint = {"classy_state_dict": {layername:value}}
      #   2. "" - if the model_file is not a nested dictionary for model weights i.e.
      #      checkpoint = {layername:value}
      #   3. key name that your model checkpoint uses for state_dict key name.
      #      checkpoint = {"your_key_name": {layername:value}}
      STATE_DICT_KEY_NAME: "classy_state_dict"
      # specify what layer should not be loaded. Layer names with this key are not copied
      # By default, set to BatchNorm stats "num_batches_tracked" to be skipped.
      SKIP_LAYERS: ["num_batches_tracked"]
      ####### If loading a non-VISSL trained model, set the following two args carefully #########
      # to make the checkpoint compatible with VISSL, if you need to remove some names
      # from the checkpoint keys, specify the name
      REMOVE_PREFIX: ""
      # In order to load the model (if not trained with VISSL) with VISSL, there are 2 scenarios:
      #    1. If you are interested in evaluating the model features and freeze the trunk.
      #       Set APPEND_PREFIX="trunk.base_model." This assumes that your model is compatible
      #       with the VISSL trunks. The VISSL trunks start with "_feature_blocks." prefix. If
      #       your model doesn't have these prefix you can append them. For example:
      #       For TorchVision ResNet trunk, set APPEND_PREFIX="trunk.base_model._feature_blocks."
      #    2. where you want to load the model simply and finetune the full model.
      #       Set APPEND_PREFIX="trunk."
      #       This assumes that your model is compatible with the VISSL trunks. The VISSL
      #       trunks start with "_feature_blocks." prefix. If your model doesn't have these
      #       prefix you can append them.
      #       For TorchVision ResNet trunk, set APPEND_PREFIX="trunk._feature_blocks."
      # NOTE: the prefix is appended to all the layers in the model
      APPEND_PREFIX: ""

Vary the number of gpus
~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL makes it extremely easy to vary the number of gpus to be used in training. For example: to train the RotNet model on 4 machines (32gpus)
or 1gpu, the changes required are:

* **Training on 1-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/rotnet/rotnet_8gpu_resnet config.DISTRIBUTED.NUM_PROC_PER_NODE=1


* **Training on 4 machines i.e. 32-gpu:**

.. code-block:: bash

    python tools/run_distributed_engines.py config=pretrain/rotnet/rotnet_8gpu_resnet config.DISTRIBUTED.NUM_PROC_PER_NODE=8 config.DISTRIBUTED.NUM_NODES=4


.. note::

    Please adjust the learning rate following `ImageNet in 1-Hour <https://arxiv.org/abs/1706.02677>`_ if you change the number of gpus.


Pre-trained models
--------------------
See `VISSL Model Zoo <https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md>`_ for the PyTorch pre-trained models with
VISSL using RotNet approach and the benchmarks.


Citation
---------

.. code-block:: none

    @misc{yan2019clusterfit,
      title={ClusterFit: Improving Generalization of Visual Representations},
      author={Xueting Yan and Ishan Misra and Abhinav Gupta and Deepti Ghadiyaram and Dhruv Mahajan},
      year={2019},
      eprint={1912.03330},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
