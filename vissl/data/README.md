# VISSL Datasets

VISSL allows reading data from multiple sources and in multiple formats. For example: users can specify a data folder path or users can specify the paths to files (.npy) that contain the array of image paths. VISSL uses a `dataset_catalog.json` that stores information about the datasets and how to obtain them. A dataset can be used by accessing [DatasetCatalog](https://github.com/facebookresearch/vissl/tree/master/vissl/data/dataset_catalog.json)
for its data.

## Expected dataset structure for ImageNet, Places205, Places365

```
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
```

## Expected dataset structure for Pascal VOC [2007, 2012]:
```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
  JPEGImages/
```

## Expected dataset structure for COCO2014

```
coco/
  annotations/
    instances_train2014.json
    instances_val2014.json
  train2014/
    # image files that are mentioned in the corresponding json
  val2014/
    # image files that are mentioned in the corresponding json
```

## Dataset Catalog
It contains a mapping from strings (which are names that identify a dataset,
e.g. "imagenet1k_folder") to a `dict` which contains:
1. mapping of various data splits (train, test, val) to the data source (path on the disk whether a folder path or a filelist)
2. source of the data (`disk_filelist` | `disk_folder`)
    - `disk_folder`: this is simply the root folder path to the downloaded data.
    - `disk_filelist`: These are numpy files: (1) file containing images information (2) file containing corresponding labels for images. We provide [scripts](https://github.com/facebookresearch/vissl/blob/master/extra_scripts/README.md) that can be used to prepare these two files for a dataset of choice.

The purpose of having this catalog is to make it easy to choose different datasets,
by just using the strings in the config.

### Creating a custon Dataset catalog `dataset_catalog.json`

Users can edit the template `vissl/hydra_configs/config/dataset_catalog.json` file to specify their datasets. The json file can be fully decided by user and can have any number of supported datasets (one or more). User can give the string names to dataset as per their choice.

#### Template for a dataset entry in `dataset_catalog.json`

```json
"data_name": {
     "train": [
         "<images_path_or_folder>", "<labels_path_or_folder>"
     ],
     "val": [
         "<images_path_or_folder>", "<labels_path_or_folder>"
     ],
 }
 ```

 User can mix match the source of image, labels i.e. labels can be filelist and images can be folder path. The yaml configuration files require specifying `LABEL_SOURCES` and `DATA_SOURCES` which allows the code to figure out how to ingest various sources.
