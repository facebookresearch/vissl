Using Data
=================

To use a dataset in VISSL, the only requirements are:

- the dataset name should be registered with :code:`VisslDatasetCatalog` in VISSL. Only name is important and the paths are not. The paths can be specifed in the configuration file. Users can either edit the :code:`dataset_catalog.json` or specify the paths in the configuration file.

- the dataset should be from a supported data source.


Reading data from several sources
------------------------------------------

VISSL allows reading data from multiple sources (disk, etc) and in multiple formats (a folder path, a :code:`.npy` file, or torchvision datasets).
The `GenericSSLDataset <https://github.com/facebookresearch/vissl/blob/main/vissl/data/ssl_dataset.py>`_ class is defined to support reading data from multiple data sources. For example: :code:`data = [dataset1, dataset2]` and the minibatches generated will have the corresponding data from each dataset.
For this reason, we also support labels from multiple sources. For example :code:`targets = [dataset1 targets, dataset2 targets]`.

Source of the data (:code:`disk_folder` | :code:`disk_filelist` | :code:`torchvision_dataset`):

- :code:`disk_folder`: this is simply the root folder path to the downloaded data.

- :code:`disk_filelist`: These are numpy (or .pkl) files: (1) file containing images information (2) file containing corresponding labels for images. We provide `scripts <https://github.com/facebookresearch/vissl/blob/main/extra_scripts/README.md>`_ that can be used to prepare these two files for a dataset of choice.

- :code:`torchvision_dataset`: the root folder path to the torchvision dowloaded dataset. As of now, the supported datasets are: CIFAR10, CIFAR100, MNIST, STL10 and SVHN.

To use a dataset, VISSL takes following inputs in the configuration file for each dataset split (train, test):

- :code:`DATASET_NAMES`: names of the datasets that are registered with :code:`VisslDatasetCatalog`. Registering dataset name is important. Example: :code:`DATASET_NAMES=[imagenet1k_folder, my_new_dataset_filelist]`

- :code:`DATA_SOURCES`: the sources of dataset. Options: :code:`disk_folder | disk_filelist`. This specifies where the data lives. Users can extend it for their purposes. Example :code:`DATA_SOURCES=[disk_folder, disk_filelist]`

- :code:`DATA_PATHS`: the paths to the dataset. The paths could be folder path (example Imagenet1k folder) or .npy filepaths. For the folder paths, VISSL uses :code:`ImageFolder` from PyTorch. Example :code:`DATA_PATHS=[<imagenet1k_folder_path>, <numpy_file_path_for_new_dataset>]`

- :code:`LABEL_SOURCES`: just like images, the targets can also come from several sources. Example: :code:`LABEL_SOURCES=[disk_folder]` for Imagenet1k. Example: :code:`DATA_SOURCES=[disk_folder, disk_filelist]`

- :code:`LABEL_PATHS`: similar to :code:`DATA_PATHS` but for labels. Example :code:`LABEL_PATHS=[<imagenet1k_folder_path>, <numpy_file_path_for_new_dataset_labels>]`

- :code:`LABEL_TYPE`: choose from :code:`standard | sample_index`. :code:`sample_index` is a common practice in self-supervised learning and :code:`sample_index`=id of the sample in the data. :code:`standard` label type is used for supervised learning and user specifis the annotated labels to use.

- :code:`DATA_LIMIT`: How many samples to train with per epoch. This can be useful for debugging purposes or for evaluating low-shot learning. Note that the default :code:`DATA_LIMIT=-1` uses the full dataset. You can also configure additional options, like the seed and ensuring class-balanced sampling in :code:`DATA_LIMIT_SAMPLING`.


Using :code:`dataset_catalog.json`
--------------------------------------

In order to use a dataset with VISSL, the dataset name must be registered with :code:`VisslDatasetCatalog`. VISSL maintains a `dataset_catalog.json <https://github.com/facebookresearch/vissl/blob/main/configs/config/dataset_catalog.json>`_ which is parsed by :code:`VisslDatasetCatalog` and the datasets
are registered with VISSL, ready-to-use.

Users can edit the template `dataset_catalog.json <https://github.com/facebookresearch/vissl/blob/main/configs/config/dataset_catalog.json>`_ file
to specify their datasets paths. Alternatively users can create their own dataset catalog json file and set the environment variable :code:`VISSL_DATASET_CATALOG_PATH` to its absolute path.
This may be helpful if you are not building the code from source or are actively developing on VISSL. The json file can be fully decided by user and can have any number of supported datasets (one or more).
Users can give the string names of the dataset they wish to use.

Template for a dataset entry in :code:`dataset_catalog.json`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    "data_name": {
       "train": [
         "<images_path_or_folder>", "<labels_path_or_folder>"
       ],
       "val": [
         "<images_path_or_folder>", "<labels_path_or_folder>"
       ],
    }


The :code:`images_path_or_folder` and :code:`labels_path_or_folder` can be directories or filepaths (numpy, pickle.)

User can mix and match the source of image, labels. i.e. labels can be filelist and images can be folder path. The yaml configuration files require specifying :code:`LABEL_SOURCES` and :code:`DATA_SOURCES` which allows the code to figure out how to ingest various sources.

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

The expected format is the exact same format used by torchvision, and the exact format obtained after either:

- expanding the "CIFAR-10 python version" archive available at https://www.cs.toronto.edu/~kriz/cifar.html

- instantiating the :code:`torchvision.datasets.CIFAR10` class with :code:`download=True`

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

The expected format is the exact same format used by torchvision, and the exact format obtained after either:

- expanding the "CIFAR-100 python version" archive available at https://www.cs.toronto.edu/~kriz/cifar.html

- instantiating the :code:`torchvision.datasets.CIFAR100` class with :code:`download=True`

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

The expected format is the exact same format used by torchvision, and the exact format obtained after either:

- expanding the :code:`stl10_binary.tar.gz` archive available at https://cs.stanford.edu/~acoates/stl10/

- instantiating the :code:`torchvision.datasets.STL10` class with :code:`download=True`

.. code-block::

    stl10_binary/
        class_names.txt
        fold_indices.txt
        test_X.bin
        test_y.bin
        train_X.bin
        train_y.bin
        unlabeled_X.bin


Expected dataset structure for SVHN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected format is the exact same format used by torchvision, and the exact format obtained after either:

- downloading the :code:`train_32x32.mat`, :code:`test_32x32.mat` and :code:`extra_32x32.mat` files available at http://ufldl.stanford.edu/housenumbers/ in the same folder

- instantiating the :code:`torchvision.datasets.SVHN` class with :code:`download=True`

.. code-block::

    svhn_folder/
        test_32x32.mat
        train_32x32.mat


Expected dataset structure for the other benchmark datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VISSL supports benchmarks inspired by the `VTAB <https://arxiv.org/pdf/1910.04867.pdf>`_ and `CLIP <https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf>`_ papers, for which the datasets either:

- Do not directly exist but are transformations of existing dataset (like images extracted from videos)
- Are not in a format directly compatible with the :code:`disk_folder` or the :code:`disk_filelist` format of VISSL
- And are not yet part of `torchvision <https://pytorch.org/vision/stable/datasets.html>`_ datasets

To run these benchmarks, the following data preparation scripts are mandatory:

- :code:`extra_scripts/datasets/create_caltech101_data_files.py`: to transform the `Caltech101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ dataset to the :code:`disk_folder` format
- :code:`extra_scripts/datasets/create_clevr_count_data_files.py`: to create a :code:`disk_filelist` dataset from `CLEVR <https://cs.stanford.edu/people/jcjohns/clevr/>`_ where the goal is to count the number of object in the scene
- :code:`extra_scripts/datasets/create_clevr_dist_data_files.py`: to create a :code:`disk_filelist` dataset from `CLEVR <https://cs.stanford.edu/people/jcjohns/clevr/>`_ where the goal is to estimate the distance of the closest object in the scene
- :code:`extra_scripts/datasets/create_dsprites_location_data_files.py`: to create a :code:`disk_folder` dataset from `dSprites <https://github.com/deepmind/dsprites-dataset>`_ where the goal is to estimate the x coordinate of the sprite on the scene
- :code:`extra_scripts/datasets/create_dsprites_orientation_data_files.py`: to create a :code:`disk_folder` dataset from `dSprites <https://github.com/deepmind/dsprites-dataset>`_ where the goal is to estimate the orientation of the sprite on the scene
- :code:`extra_scripts/datasets/create_dtd_data_files.py`: to transform the `DTD <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_ dataset to the :code:`disk_folder` format
- :code:`extra_scripts/datasets/create_euro_sat_data_files.py`: to transform the `EUROSAT <https://github.com/phelber/eurosat>`_ dataset to the :code:`disk_folder` format
- :code:`extra_scripts/datasets/create_fgvc_aircraft_data_files.py`: to transform the `FGVC Aircrafts <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ dataset to the :code:`disk_folder` format
- :code:`extra_scripts/datasets/create_food101_data_files.py`: to transform the `FOOD101 <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101>`_ dataset to the :code:`disk_folder` format
- :code:`extra_scripts/datasets/create_gtsrb_data_files.py`: to transform the `GTSRB <https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html>`_ dataset to the :code:`disk_folder` format
- :code:`extra_scripts/datasets/create_imagenet_ood_data_files.py`: to create test sets in :code:`disk_filelist` format for Imagenet based on `Imagenet-A <https://gith`ub.com/hendrycks/natural-adv-examples>`_ and `Imagenet-R <https://github.com/hendrycks/imagenet-r>`_
- :code:`extra_scripts/datasets/create_kitti_dist_data_files.py`: to create a :code:`disk_folder` dataset from `KITTI <http://www.cvlibs.net/datasets/kitti/>`_ where the goal is to estimate the distance of the closest car, van or truck
- :code:`extra_scripts/datasets/create_oxford_pets_data_files.py`: to transform the `Oxford Pets <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ dataset to the :code:`disk_folder` format
- :code:`extra_scripts/datasets/create_patch_camelyon_data_files.py`: to transform the `PatchCamelyon <https://github.com/basveeling/pcam>`_ dataset to the :code:`disk_folder` format
- :code:`extra_scripts/datasets/create_small_norb_azimuth_data_files.py` to create a :code:`disk_folder` dataset from `Small NORB <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/>`_ where the goal is to find the azimuth or the photographed object
- :code:`extra_scripts/datasets/create_small_norb_elevation_data_files.py` to create a :code:`disk_folder` dataset from `Small NORB <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/>`_ where the goal is to predict the elevation in the image
- :code:`extra_scripts/datasets/create_sun397_data_files.py` to transform the `SUN397 <https://vision.princeton.edu/projects/2010/SUN/>`_ dataset to the :code:`disk_filelist` format
- :code:`extra_scripts/datasets/create_ucf101_data_files.py`: to create a :code:`disk_folder` image action recognition dataset from the video action recognition dataset `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ by extracting the middle frame

You can read more about how to download these datasets and run these scripts from `here <https://github.com/facebookresearch/vissl/blob/main/extra_scripts/README.md>`_.

After data preparation, the output folders are either compatible with the :code:`disk_filelist` layout:

.. code-block:: bash

    train_images.npy  # Paths to the train images
    train_labels.npy  # Labels for each of the train images
    val_images.npy    # Paths to the val images
    val_labels.npy    # Labels for each of the val images

Or with the :code:`disk_folder` layout:

.. code-block:: bash

    train/
        label1/
            image_1.jpeg
            image_2.jpeg
            ...
        label2/
            image_x.jpeg
            image_y.jpeg
            ...
        ...
    val/
        label1/
            image_1.jpeg
            image_2.jpeg
            ...
        label2/
            image_x.jpeg
            image_y.jpeg
            ...
        ...

.. note::

    In the case of the :code:`disk_folder` layout, the images are copied into the output folder and the input folder is not necessary anymore.
    The input folder can for instance be deleted.

    In the case of the :code:`disk_filelist` layout, the images are referenced inside the :code:`.npy` files.
    It is therefore important to keep the input folder and not alter it (which includes not moving it).

    The :code:`disk_filelist` has the advantage of using less space, while the :code:`disk_folder` offers total decoupling from the
    original dataset files and is more advantageous for small number of images or when the inputs do not allow to reference images
    (for instance when extracting frames from videos, or dealing with images in an unsupported format).

    The aforementioned scripts use the either the :code:`disk_folder` or :code:`disk_filelist` based on these constraints.


Dataloader
------------------------------------------

VISSL uses PyTorch :code:`torch.utils.data.DataLoader` and allows setting all the dataloader option as below. The dataloader is wrapped with `DataloaderAsyncGPUWrapper <https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/dataset/dataloader_async_gpu_wrapper.py>`_ or `DataloaderSyncGPUWrapper <https://github.com/facebookresearch/vissl/blob/main/vissl/data/dataloader_sync_gpu_wrapper.py>`_ depending on whether user wants to copy data to gpu async or not.

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

VISSL supports PyTorch default collator :code:`torch.utils.data.dataloader.default_collate` and also many custom data collators used in self-supervision. The use any collator, user has to simply specify the :code:`DATA.TRAIN.COLLATE_FUNCTION` to be the name of the collator to use. See all custom VISSL collators implemented `here <https://github.com/facebookresearch/vissl/tree/main/vissl/data/collators>`_.

An example for specifying collator for SwAV training:

.. code-block:: yaml

    DATA:
      TRAIN:
        COLLATE_FUNCTION: multicrop_collator


Using Data Transforms
------------------------------------------

VISSL supports all PyTorch :code:`TorchVision` transforms, `Augly Transforms <https://github.com/facebookresearch/AugLy>`_, as well as many custom transforms required by Self-supervised approaches including MoCo, SwAV, PIRL, SimCLR, BYOL, etc. Using Transforms is intuitive and easy in VISSL. Users specify the list of transforms they want to apply on the data in the order of application.
This involves using the transform name and the key:value to specify the parameter values for the transform. See the full list of transforms implemented by VISSL `here <https://github.com/facebookresearch/vissl/tree/main/vissl/data/ssl_transforms>`_

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

To use an `Augly Transforms <https://github.com/facebookresearch/AugLy>`_, please first :code:`pip install augly` and use :code:`>= Python 3.7 `. Then specify the Augly transform class name, set :code:`transform_type: augly`, and specify any other arguments required. For example using the `Overlay Emoji Transform <https://github.com/facebookresearch/AugLy/blob/263284f4a8b5b76d457005ed27298ed0b4e6b362/augly/image/transforms.py#L1029>`_:

.. code-block:: yaml

    DATA:
      TRAIN:
        TRANSFORMS:
          - name: OverlayEmoji
            transform_type: augly
            opacity: 0.5
            emoji_size: 0.2
            x_pos: 0.3
            y_pos: 0.4
            p: 0.7


Using Data Sampler
------------------------------------------

VISSL supports 2 types of samplers:

- PyTorch default :code:`torch.utils.data.distributed.DistributedSampler`

- VISSL sampler `StatefulDistributedSampler <https://github.com/facebookresearch/vissl/blob/main/vissl/data/data_helper.py>`_ that is written specifically for large scale dataset trainings. See the `documentation <https://github.com/facebookresearch/vissl/blob/main/vissl/data/data_helper.py>`_ for the sampler.


By default, the PyTorch default sampler is used unless user specifies :code:`DATA.TRAIN.USE_STATEFUL_DISTRIBUTED_SAMPLER=true` in which case :code:`StatefulDistributedSampler` will be used.
