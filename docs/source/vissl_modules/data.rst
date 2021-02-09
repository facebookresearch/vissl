Using Data
=================

To use a dataset in VISSL, the only requirements are:

- the dataset name should be registered with :code:`VisslDatasetCatalog` in VISSL. Only name is important and the paths are not. The paths can be specifed in the configuration file. Users can either edit the :code:`dataset_catalog.json` or specify the paths in the configuration file.

- the dataset should be from a supported data source.


Reading data from several sources
------------------------------------------

VISSL allows reading data from multiple sources (disk, etc) and in multiple formats (a folder path, a :code:`.npy` file, or torchvision datasets).
The `GenericSSLDataset <https://github.com/facebookresearch/vissl/blob/master/vissl/data/ssl_dataset.py>`_ class is defined to support reading data from multiple data sources. For example: :code:`data = [dataset1, dataset2]` and the minibatches generated will have the corresponding data from each dataset.
For this reason, we also support labels from multiple sources. For example :code:`targets = [dataset1 targets, dataset2 targets]`.

Source of the data (:code:`disk_filelist` | :code:`disk_folder` | :code:`torchvision_dataset`):

- :code:`disk_folder`: this is simply the root folder path to the downloaded data.

- :code:`disk_filelist`: These are numpy (or .pkl) files: (1) file containing images information (2) file containing corresponding labels for images. We provide `scripts <https://github.com/facebookresearch/vissl/blob/master/extra_scripts/README.md>`_ that can be used to prepare these two files for a dataset of choice.

- :code:`torchvision_dataset`: the root folder path to the torchvision dowloaded dataset. As of now, the supported datasets are: CIFAR-10, CIFAR-100, MNIST and STL-10.

To use a dataset, VISSL takes following inputs in the configuration file for each dataset split (train, test):

- :code:`DATASET_NAMES`: names of the datasets that are registered with :code:`VisslDatasetCatalog`. Registering dataset name is important. Example: :code:`DATASET_NAMES=[imagenet1k_folder, my_new_dataset_filelist]`

- :code:`DATA_SOURCES`: the sources of dataset. Options: :code:`disk_folder | disk_filelist`. This specifies where the data lives. Users can extend it for their purposes. Example :code:`DATA_SOURCES=[disk_folder, disk_filelist]`

- :code:`DATA_PATHS`: the paths to the dataset. The paths could be folder path (example Imagenet1k folder) or .npy filepaths. For the folder paths, VISSL uses :code:`ImageFolder` from PyTorch. Example :code:`DATA_PATHS=[<imagenet1k_folder_path>, <numpy_file_path_for_new_dataset>]`

- :code:`LABEL_SOURCES`: just like images, the targets can also come from several sources. Example: :code:`LABEL_SOURCES=[disk_folder]` for Imagenet1k. Example: :code:`DATA_SOURCES=[disk_folder, disk_filelist]`

- :code:`LABEL_PATHS`: similar to :code:`DATA_PATHS` but for labels. Example :code:`LABEL_PATHS=[<imagenet1k_folder_path>, <numpy_file_path_for_new_dataset_labels>]`

- :code:`LABEL_TYPE`: choose from :code:`standard | sample_index`. :code:`sample_index` is a common practice in self-supervised learning and :code:`sample_index`=id of the sample in the data. :code:`standard` label type is used for supervised learning and user specifis the annotated labels to use.


Using :code:`dataset_catalog.json`
--------------------------------------

In order to use a dataset with VISSL, the dataset name must be registered with :code:`VisslDatasetCatalog`. VISSL maintains a `dataset_catalog.json <https://github.com/facebookresearch/vissl/blob/master/configs/config/dataset_catalog.json>`_ which is parsed by :code:`VisslDatasetCatalog` and the datasets
are registered with VISSL, ready-to-use.

Users can edit the template `dataset_catalog.json <https://github.com/facebookresearch/vissl/blob/master/configs/config/dataset_catalog.json>`_ file
to specify their datasets paths. The json file can be fully decided by user and can have any number of supported datasets (one or more). User can give the string names to dataset as per their choice.

Template for a dataset entry in :code:`dataset_catalog.json`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    "data_name": {
       "train": [
         "<images_path_or_folder>", "<labels_path_or_folder>"
       ],
       "val": [
         "<images_path_or_folder>", "<labels_path_or_folder>"
       ],
    }


The :code:`images_path_or_folder` and :code:`labels_path_or_folder` can be directories or filepaths (numpy, pickle.)

User can mix match the source of image, labels i.e. labels can be filelist and images can be folder path. The yaml configuration files require specifying :code:`LABEL_SOURCES` and :code:`DATA_SOURCES` which allows the code to figure out how to ingest various sources.

.. note::

    Filling the :code:`dataset_catalog.json` is a one time process only and provides the benefits of simply accessing any dataset with the dataset name in the configuration files for the rest of the trainings.


Using Builtin datasets
------------------------

VISSL supports several Builtin datasets as indicated in the :code:`dataset_catalog.json` file. Users can specify paths to those datasets.

Expected dataset structure for ImageNet, Places205, Places365
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    {imagenet, places205, places365}
    train/
        <n0......>/
            <im-1-name>.JPEG
            ...
            <im-N-name>.JPEG
            ...
        <n1......>/
            <im-1-name>.JPEG
            ...
            <im-M-name>.JPEG
            ...
        ...
    val/
        <n0......>/
            <im-1-name>.JPEG
            ...
            <im-N-name>.JPEG
            ...
        <n1......>/
            <im-1-name>.JPEG
            ...
            <im-M-name>.JPEG
            ...
        ...


Expected dataset structure for Pascal VOC [2007, 2012]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    VOC20{07,12}/
        Annotations/
        ImageSets/
            Main/
            trainval.txt
            test.txt
        JPEGImages/


Expected dataset structure for COCO2014
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    coco/
        annotations/
            instances_train2014.json
            instances_val2014.json
        train2014/
            # image files that are mentioned in the corresponding json
        val2014/
            # image files that are mentioned in the corresponding json


Expected dataset structure for CIFAR10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected format is the exact same format used by torchvision, and the exact format obtained after expanding the
"CIFAR-10 python version" archive available at https://www.cs.toronto.edu/~kriz/cifar.html.

.. code-block::

    cifar-10-batches-py/
        batches.meta
        data_batch_1
        data_batch_2
        data_batch_3
        data_batch_4
        data_batch_5
        readme.html
        test_batch


Expected dataset structure for CIFAR100
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected format is the exact same format used by torchvision, and the exact format obtained after expanding the
"CIFAR-100 python version" archive available at https://www.cs.toronto.edu/~kriz/cifar.html.*

.. code-block::

    cifar-100-python/
        meta
        test
        train


Expected dataset structure for MNIST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected format is the exact same format used by torchvision, and the exact format obtained after
instantiating the :code:`torchvision.datasets.MNIST` class with the flag :code:`download=True`.

.. code-block::

    MNIST/
        processed/
            test.pt
            training.pt
        raw/
            t10k-images-idx3-ubyte
            t10k-images-idx3-ubyte.gz
            t10k-labels-idx1-ubyte
            t10k-labels-idx1-ubyte.gz
            train-images-idx3-ubyte
            train-images-idx3-ubyte.gz
            train-labels-idx1-ubyte
            train-labels-idx1-ubyte.gz


Expected dataset structure for STL10
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected format is the exact same format used by torchvision, and the exact format obtained after expanding the
:code:`stl10_binary.tar.gz` archive available at https://academictorrents.com/details/a799a2845ac29a66c07cf74e2a2838b6c5698a6a.

.. code-block::

    stl10_binary/
        class_names.txt
        fold_indices.txt
        test_X.bin
        test_y.bin
        train_X.bin
        train_y.bin
        unlabeled_X.bin


Dataloader
------------------------------------------

VISSL uses PyTorch :code:`torch.utils.data.DataLoader` and allows setting all the dataloader option as below. The dataloader is wrapped with `DataloaderAsyncGPUWrapper <https://github.com/facebookresearch/ClassyVision/blob/master/classy_vision/dataset/dataloader_async_gpu_wrapper.py>`_ or `DataloaderSyncGPUWrapper <https://github.com/facebookresearch/vissl/blob/master/vissl/data/dataloader_sync_gpu_wrapper.py>`_ depending on whether user wants to copy data to gpu async or not.

The settings for the :code:`Dataloader` in VISSL are:

.. code-block:: bash

    dataset (GenericSSLDataset):    the dataset object for which dataloader is constructed
    dataset_config (dict):          configuration of the dataset. it should be DATA.TRAIN or DATA.TEST settings
    num_dataloader_workers (int):   number of workers per gpu (or cpu) training
    pin_memory (bool):              whether to pin memory or not
    multi_processing_method (str):  method to use. options: forkserver | fork | spawn
    device (torch.device):          training on cuda or cpu
    get_sampler (get_sampler):      function that is used to get the sampler
    worker_init_fn (None default):  any function that should be executed during initialization of dataloader workers


Using Data Collators
------------------------------------------

VISSL supports PyTorch default collator :code:`torch.utils.data.dataloader.default_collate` and also many custom data collators used in self-supervision. The use any collator, user has to simply specify the :code:`DATA.TRAIN.COLLATE_FUNCTION` to be the name of the collator to use. See all custom VISSL collators implemented `here <https://github.com/facebookresearch/vissl/tree/master/vissl/data/collators>`_.

An example for specifying collator for SwAV training:

.. code-block:: yaml

    DATA:
      TRAIN:
        COLLATE_FUNCTION: multicrop_collator


Using Data Transforms
------------------------------------------

VISSL supports all PyTorch :code:`TorchVision` transforms as well as many transforms required by Self-supervised approaches including MoCo, SwAV, PIRL, SimCLR, BYOL, etc. Using Transforms is very intuitive and easy in VISSL. Users specify the list of transforms they want to apply on the data in the order of application.
This involves using the transform name and the key:value to specify the parameter values for the transform. See the full list of transforms implemented by VISSL `here <https://github.com/facebookresearch/vissl/tree/master/vissl/data/ssl_transforms>`_

An example of transform for SwAV:

.. code-block:: yaml

    DATA:
      TRAIN:
        TRANSFORMS:
          - name: ImgPilToMultiCrop
            total_num_crops: 6
            size_crops: [224, 96]
            num_crops: [2, 4]
            crop_scales: [[0.14, 1], [0.05, 0.14]]
          - name: RandomHorizontalFlip
            p: 0.5
          - name: ImgPilColorDistortion
            strength: 1.0
          - name: ImgPilGaussianBlur
            p: 0.5
            radius_min: 0.1
            radius_max: 2.0
          - name: ToTensor
          - name: Normalize
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]




Using Data Sampler
------------------------------------------

VISSL supports 2 types of samplers:

- PyTorch default :code:`torch.utils.data.distributed.DistributedSampler`

- VISSL sampler `StatefulDistributedSampler <https://github.com/facebookresearch/vissl/blob/master/vissl/data/data_helper.py>`_ that is written specifically for large scale dataset trainings. See the documentation for the sampler.


By default, the PyTorch default sampler is used unless user specifies :code:`DATA.TRAIN.USE_STATEFUL_DISTRIBUTED_SAMPLER=true` in which case :code:`StatefulDistributedSampler` will be used.
