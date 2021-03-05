Add new Data Source
=======================

VISSL supports data loading from :code:`disk_folder` as the default data source. If your dataset lives in a custom data storage solution instead of :code:`disk_folder`, you can extend VISSL to work with your data storage in several different ways:

- Exporting the content of the non supported datasource as :code:`disk_folder`
- Implementing a new type of data source inside VISSL

The two options are developed below.

Transforming to a disk_folder format
---------------------------------------

Out of the box, VISSL supports any dataset following the :code:`disk_folder` format:

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

The transformation of most existing dataset to the :code:`disk_folder` should be relatively easy. Some example scripts are provided in the folder `extra_script <https://github.com/facebookresearch/vissl/tree/master/extra_scripts>`_ demonstrating how to perform this transformation. For instance:

- :code:`create_clevr_count_data_files.py`: to create a dataset from `CLEVR <https://arxiv.org/abs/1612.068901>`_ where the goal is to count the number of object in the scene
- :code:`create_ucf101_data_files.py`: to create an image action recognition dataset from the video action recognition dataset `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ by extracting the middle frame

You can refer to the documentation available `here <https://github.com/facebookresearch/vissl/blob/master/extra_scripts/README.md>`_ to get more information all the available data preparation scripts.

Once a dataset is made available at path :code:`/path/to/dataset`, plugging the dataset in the library is a simple two step process:

1. add the paths to the "dataset_catalog.json" registry of dataset

.. code-block:: json

    "my_dataset_folder": {
        "train": ["/path/to/dataset/train", "<lbl_path>"],
        "val": ["/path/to/dataset/val", "<lbl_path>"]
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


Adding a new type of data source
------------------------------------

If instead, you want to use a custom data storage solution :code:`my_data_source` instead of :code:`disk_folder`, you can extend VISSL to work with the :code:`my_data_source` data storage by following the steps below:

- **Step1**: Implement your custom data source under :code:`vissl/data/my_data_source.py` following the template:

.. code-block:: bash

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

  - See our documentation on "Using dataset_catalog.json" to update the :code:`configs/dataset_catalog.json` file.

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
