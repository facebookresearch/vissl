Instance Retrieval and Copy detection Benchmarks
===================================================

It has been shown that self-supervised models have `state-of-the art <https://arxiv.org/abs/2104.14294>`_ performance on Instance Retrieval and Copy Detection. VISSL supports benchmarking for the following datasets: `ROxford <https://arxiv.org/abs/1803.11285>`_, `RParis <https://arxiv.org/abs/1803.11285>`_, and `CopyDays <https://lear.inrialpes.fr/~jegou/data.php>`_.

Setting Up Datasets
------------------------------------------

To setup the datasets, please follow the steps below.


Revisited Oxford/Paris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These datasets test instance retrieval for landmarks in Oxford and Paris and are an update of the original Oxford/Paris datasets. For more information about this dataset see `here <http://cmp.felk.cvut.cz/revisitop/>`_.

To setup the datasets, we have convenience scripts for `RParis <https://github.com/facebookresearch/vissl/blob/main/extra_scripts/datasets/create_rparis_dataset.py>`_ and `ROxford <https://github.com/facebookresearch/vissl/blob/main/extra_scripts/datasets/create_roxford_dataset.py>`_. For example:

.. code-block:: bash

    python extra_scripts/datasets/create_oxford_dataset.py
        -i /path/to/roxford/
        -o /output_path/roxford
        -d


CopyDays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These datasets test copy detection performance. For more information about this dataset see `here <https://lear.inrialpes.fr/~jegou/data.php/>`_.

To setup the dataset, please setup the datasets according to these `instructions<https://lear.inrialpes.fr/~jegou/data.php>`_. For example:


Evaluating the Datasets
------------------------------------------

At a high level, the features for the database, query, and train images are extracted as follows:

- **Step1**: Images are loaded, resized, normalized, and converted to a Tensor.

- **Step2**: Images are fed to the model. Optionally you can feed the image to the model with multiple scalings, using :code:`IMG_SCALINGS=[1, 0.5]` for example.

- **Step3 (Optional)**: Post-processing is performed on the model output. You can use no post-processing, `gem <https://arxiv.org/pdf/1711.02512.pdf>`_, or `rmac <https://arxiv.org/pdf/1511.05879.pdf>`_. This option is controlled by :code:`FEATS_PROCESSING_TYPE`.

- **Step4 (Optional)**: Normalize the features. This option is controlled by :code:`NORMALIZE_FEATURES`.

The entire evaluation is as follows:

- **Step1 (Optional)**: Extract the train features as above and train PCA on the features. You must set :code:`EVAL_DATASET_NAME` and :code:`TRAIN_PCA_WHITENING: True`.

- **Step2**: Extract the database and query features as above.

- **Step3 (Optional)**: Optionally apply the PCA fit in step-1 to the database and query images.

- **Step4**: For each query image, rank the database according to the :code:`SIMILARITY_MEASURE`.

- **Step5**: Evaluate based on the Mean Average Precision metric.

We offer several configs for evaluating these datasets. See `eval_resnet_1gpu_roxford.yaml <https://github.com/facebookresearch/vissl/blob/main/configs/config/benchmark/instance_retrieval/eval_resnet_1gpu_roxford.yaml>`_, `eval_resnet_1gpu_rparis.yaml <https://github.com/facebookresearch/vissl/blob/main/configs/config/benchmark/instance_retrieval/eval_resnet_1gpu_rparis.yaml>`_, and `eval_resnet_1gpu_roxford.yaml <https://github.com/facebookresearch/vissl/blob/main/configs/config/benchmark/instance_retrieval/eval_resnet_1gpu_copydays.yaml>`_

Here is an example of the config options:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

  MODEL:
    FEATURE_EVAL_SETTINGS:
      EVAL_MODE_ON: True
      FREEZE_TRUNK_ONLY: True
      EXTRACT_TRUNK_FEATURES_ONLY: True
      SHOULD_FLATTEN_FEATS: false
      # Which feature layer to use to evaluate.
      LINEAR_EVAL_FEAT_POOL_OPS_MAP: [
          ["res5", ["Identity", []]],
      ]
  IMG_RETRIEVAL:
    ########################## Dataset Information #############################
    TRAIN_DATASET_NAME: roxford5k
    EVAL_DATASET_NAME: rparis6k
    DATASET_PATH: <enter dataset path>
    # valid only if we are training whitening on the whitening dataset
    WHITEN_IMG_LIST: ""
    # Path to the compute_ap binary to evaluate Oxford / Paris
    EVAL_BINARY_PATH: ""
    # Sets data limits for the number of training, query, and database samples.
    DEBUG_MODE: False
    # Number of training samples to use. -1 uses all the samples in the dataset.
    NUM_TRAINING_SAMPLES: -1
    # Number of query samples to use. -1 uses all the samples in the dataset.
    NUM_QUERY_SAMPLES: -1
    # Number of database samples to use. -1 uses all the samples in the dataset.
    NUM_DATABASE_SAMPLES: -1
    # Whether or not to use distractor images. Distractors should be under DATASET_PATH/distractors dir.
    USE_DISTRACTORS: False
    # IMG_SCALINGS=List[int], where features are extracted for each
    # image scale and averaged. Default is [1], meaning that only the full
    # image is processed.
    IMG_SCALINGS: [1]
    # cosine_similarity | l2.
    SIMILARITY_MEASURE: cosine_similarity
    ######################## Features Processing Hypers #######################
    # Resize larger side of image to RESIZE_IMG pixel
    RESIZE_IMG: 1024
    # RMAC spatial levels. See https://arxiv.org/pdf/1511.05879.pdf.
    SPATIAL_LEVELS: 3
    # output dimension of PCA
    N_PCA: 512
    # Whether to apply PCA/whitening or not
    TRAIN_PCA_WHITENING: True
    # gem  | rmac | "" (no post-processing)
    FEATS_PROCESSING_TYPE: ""
    # valid only for GeM pooling of features. Note that GEM_POOL_POWER=1 equates to average pooling.
    GEM_POOL_POWER: 4.0
    # Whether or not to crop the query images with the given region of interests --
    # Relevant for Oxford, Paris, ROxford, and RParis datasets.
    # Our experiments with RN-50/rmac show that ROI cropping degrades performance.
    CROP_QUERY_ROI: False
    # Whether or not to apply L2 norm after the features have been post-processed.
    # Normalization is heavily recommended based on experiments run.
    NORMALIZE_FEATURES: True
    ######################## Misc #######################
    # Whether or not to save the retrieval ranking scores (metrics, rankings, similarity scores)
    SAVE_RETRIEVAL_RANKINGS_SCORES: True
    # Whether or not to save the features that were extracted
    SAVE_FEATURES: False

Evaluating ROxford
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python tools/run_distributed_engines.py config=benchmark/instance_retrieval/eval_resnet_1gpu_roxford.yaml


Evaluating RParis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python tools/run_distributed_engines.py config=benchmark/instance_retrieval/eval_resnet_1gpu_rparis.yaml


Evaluating Copydays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python tools/run_distributed_engines.py config=benchmark/instance_retrieval/eval_resnet_1gpu_copydays.yaml
