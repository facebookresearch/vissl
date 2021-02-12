Add new Data Source
=======================

VISSL supports data loading from :code:`disk` as the default data source. If users dataset lives in their custom data storage solution :code:`my_data_source` instead of :code:`disk`, then users can extend VISSL to work with their data storage. Follow the steps below:

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
