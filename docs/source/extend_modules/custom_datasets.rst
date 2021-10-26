Using Custom Datasets
=========================

VISSL allows adding custom datasets easily. Using a new custom dataset has 2 requirements:

- **Requirement1**: The dataset name must be registered with :code:`VisslDatasetCatalog`.

- **Requirement2**: Users should ensure that the data source is supported by VISSL. By default, VISSL supports reading data from disk. If user data is loaded from a different data source, please add the new data source following the documentation.


Follow the steps below to register and use the new dataset:

- **Step1**: Register the dataset with VISSL. Given user dataset with dataset name :code:`my_new_dataset_name` and path to the dataset train and test splits, users can register the dataset following:


.. code-block:: python

    from vissl.data.dataset_catalog import VisslDatasetCatalog

    VisslDatasetCatalog.register_data(name="my_new_dataset_name", data_dict={"train": ... , "test": ...})


.. note::

    VISSL also supports registering the dataset via a custom :code:`json` file or or registering a python :code:`dict` with your datasets. Please see our documentation on :ref:`Using dataset_catalog.json<Using Data>`

- **Step2 (Optional)**: If the dataset requires a new data source other than disk or supported disk formats (:code:`disk_folder` or :code:`disk_filelist`), please add the new data source to VISSL.
  Follow our documentation on Adding new dataset.

- **Step3**: Test your dataset

.. code-block:: yaml

    DATA:
      TRAIN:
        DATA_SOURCES: [my_data_source]
        DATASET_NAMES: [my_new_dataset_name]
