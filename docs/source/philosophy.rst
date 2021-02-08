VISSL Philosophy
===========================

Dataset support
---------------------------

The VISSL philosophy in terms of dataset integration is:

- Out-of-the box support of torchvision dataset: all classification datasets in torchvision should be supported in VISSL
- Support of additional classification datasets through the "disk_folder" abstraction: custom datasets should be transformed into this format

Definition of the "disk_folder" format:

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

Once a dataset is made available at path "/path/to/dataset", plugging the dataset in the library is a simple two step process:

1. add the paths to the "dataset_catalog.json" registry of dataset

.. code-block:: json

    "ucf101_folder": {
        "train": ["/path/to/dataset/ucf101/train", "<lbl_path>"],
        "val": ["/path/to/dataset/ucf101/val", "<lbl_path>"]
    },

2. reference the new dataset "ucf101_folder" in a configuration file:

.. code-block:: yaml

    config:
      DATA:
        TRAIN:
          DATA_SOURCES: [disk_folder]
          LABEL_SOURCES: [disk_folder]
          DATASET_NAMES: [ucf101_folder]
          ...
        TEST:
          DATA_SOURCES: [disk_folder]
          LABEL_SOURCES: [disk_folder]
          DATASET_NAMES: [ucf101_folder]
          ...

The transformation of most existing dataset to the disk_folder should be relatively easy. A bunch of example scripts are already provided in the folder "extra_script" showing how to transform custom datasets to the "disk_folder" format.

- `extra_scripts/create_ucf101_data_files.py`: transforms the UCF101 video action classification dataset into a classification dataset by extracting the middle frame of the video as input image
- `extra_scripts/create_clevr_count_data_files.py`: transforms the CLEVR dataset into a classification dataset in which the goal is to count the number of objects appearing in the image
