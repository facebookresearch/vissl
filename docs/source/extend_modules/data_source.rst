Add new Data Source
=======================

VISSL supports data loading from :code:`disk_folder`, :code:`disk_filelist` or :code:`torchvision_dataset` as default data sources.
If your dataset lives in a custom data storage solution instead, you can extend VISSL to work with your data storage in several different ways:

- Exporting the content of the non supported datasource as :code:`disk_filelist`
- Exporting the content of the non supported datasource as :code:`disk_folder`
- Implementing a new type of data source inside VISSL

Each of these options is developed below.

Transforming to a disk_filelist format
----------------------------------------

Out of the box, VISSL supports any dataset following the :code:`disk_filelist` format:

.. code-block:: bash

    /path/to/dataset/
        train_images.npy
        train_labels.npy
        val_images.npy
        val_labels.npy

.. note::

    The name and number of partitions may differ: you can for instance create 3 different partitions for train/val/test.

The :code:`*_images.npy` files should contain the path to the images, one path for each sample, while the :code:`*_labels.npy` files should contain the corresponding labels.
There are two formats supported for the labels: either integers (from 0 to N-1 for N classes) or strings.

Once a :code:`disk_filelist` dataset is made available at path :code:`/path/to/dataset`, plugging the dataset in the library is a simple two step process:

1. add the paths to the "dataset_catalog.json" registry of dataset

.. code-block:: json

    "my_dataset_filelist": {
        "train": ["/path/to/dataset/train_images.npy", "/path/to/dataset/train_labels.npy"],
        "val": ["/path/to/dataset/val_images.npy", "/path/to/dataset/val_labels.npy"],
    },

2. reference the new dataset in a configuration file:

.. code-block:: yaml

    config:
      DATA:
        TRAIN:
          DATA_SOURCES: [disk_filelist]
          LABEL_SOURCES: [disk_filelist]
          DATASET_NAMES: [my_dataset_filelist]
          ...
        TEST:
          DATA_SOURCES: [disk_filelist]
          LABEL_SOURCES: [disk_filelist]
          DATASET_NAMES: [my_dataset_filelist]
          ...

Some examples of scripts transforming existing data sources to the :code:`disk_filelist` format can be found in the `extra_script <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_ folder.
For example, :code:`create_clever_count_data_files.py` creates a new classification dataset from the `CLEVR <https://cs.stanford.edu/people/jcjohns/clevr/>`_ dataset, in which the goal is to count the number of object in the scene.

Please refer to the documentation available `here <https://github.com/facebookresearch/vissl/blob/main/extra_scripts/README.md>`_ to get more information all the available data preparation scripts.


Transforming to a disk_folder format
---------------------------------------

Out of the box, VISSL also supports any dataset following the :code:`disk_folder` format:

.. code-block:: bash

    /path/to/dataset/
      train/
        class1/
            a.jpg
            b.jpg
            ...
        class2/
            c.jpg
            ...
      val/
        class1/
            d.jpg
            e.jpg
            ...
        class2/
            f.jpg
            ...

This format requires to copy the images, which might take more disk space than the :code:`disk_filelist` format, but is nevertheless the best option in many cases.

In particular, if the original dataset does not allow us to reference image paths (it might be a video dataset or a custom binary format), the :code:`disk_filelist` is not an option anymore and :code:`disk_folder` might be the best option.

Once a :code:`disk_folder` dataset is made available at path :code:`/path/to/dataset`, plugging the dataset in the library is a simple two step process:

1. add the paths to the "dataset_catalog.json" registry of dataset

.. code-block:: json

    "my_dataset_folder": {
        "train": ["/path/to/dataset/train", "<ignored>"],
        "val": ["/path/to/dataset/val", "<ignored>"]
    },

2. reference the new dataset in a configuration file:

.. code-block:: yaml

    config:
      DATA:
        TRAIN:
          DATA_SOURCES: [disk_folder]
          LABEL_SOURCES: [disk_folder]
          DATASET_NAMES: [my_dataset_folder]
          ...
        TEST:
          DATA_SOURCES: [disk_folder]
          LABEL_SOURCES: [disk_folder]
          DATASET_NAMES: [my_dataset_folder]
          ...

Some examples of scripts transforming existing data sources to the :code:`disk_folder` format can be found in the `extra_script <https://github.com/facebookresearch/vissl/tree/main/extra_scripts>`_ folder.
For example, :code:`create_ucf101_data_files.py`: creates an image action recognition dataset from the video action recognition dataset `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ by extracting the middle frame of each video.

Please refer to the documentation available `here <https://github.com/facebookresearch/vissl/blob/main/extra_scripts/README.md>`_ to get more information all the available data preparation scripts.

Adding a new type of data source
------------------------------------

If instead, you want to use a custom data storage solution :code:`my_data_source` instead of :code:`disk_folder`, you can extend VISSL to work with the :code:`my_data_source` data storage by following the steps below:

- **Step1**: Implement your custom data source under :code:`vissl/data/my_data_source.py` following the template:

.. code-block:: python

    from vissl.data.data_helper import get_mean_image
    from torch.utils.data import Dataset

    class MyNewSourceDataset(Dataset):
        """
        add documentation on how this dataset works

        Args:
            add docstrings for the parameters
        """

        def __init__(self, cfg, data_source, path, split, dataset_name):
            super(MyNewSourceDataset, self).__init__()
            assert data_source in [
                "disk_filelist",
                "disk_folder",
                "my_data_source"
            ], "data_source must be either disk_filelist or disk_folder or my_data_source"
            self.cfg = cfg
            self.split = split
            self.dataset_name = dataset_name
            self.data_source = data_source
            self._path = path
            # implement anything that data source init should do
            ....
            ....
            self._num_samples = ?? # set the length of the dataset


        def num_samples(self):
            """
            Size of the dataset
            """
            return self._num_samples

        def __len__(self):
            """
            Size of the dataset
            """
            return self.num_samples()

        def __getitem__(self, idx: int):
            """
            implement how to load the data corresponding to idx element in the dataset
            from your data source
            """
            ....
            ....

            # is_success should be True or False indicating whether loading data was successful or failed
            # loaded data should be Image.Image if image data
            return loaded_data, is_success


- **Step2**: Register the new data source with VISSL. Extend the :code:`DATASET_SOURCE_MAP` dict in :code:`vissl/data/__init__.py`.

.. code-block:: python

    DATASET_SOURCE_MAP = {
        "disk_filelist": DiskImageDataset,
        "disk_folder": DiskImageDataset,
        "torchvision_dataset": TorchvisionDataset,
        "synthetic": SyntheticImageDataset,
        "my_data_source": MyNewSourceDataset,
    }

- **Step3**: Register the name of the datasets you plan to load using the new data source. There are 2 ways to do this:

  - See our documentation on :ref:`Using dataset_catalog.json<Using Data>` to update the :code:`configs/dataset_catalog.json` file.

  - Insert a python call following:

    .. code-block:: bash

        # insert the following call in your python code
        from vissl.data.dataset_catalog import VisslDatasetCatalog

        VisslDatasetCatalog.register_data(name="my_dataset_name", data_dict={"train": ... , "test": ...})

- **Step4**: Test using your dataset

.. code-block:: yaml

    DATA:
      TRAIN:
        DATA_SOURCES: [my_data_source]
        DATASET_NAMES: [my_dataset_name]
