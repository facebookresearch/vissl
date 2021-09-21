# VISSL Datasets

VISSL allows reading data from multiple sources and in multiple formats. For example: users can specify a data folder path or users can specify the paths to files (.npy) that contain the array of image paths. VISSL uses a `dataset_catalog.json` that stores information about the datasets and how to obtain them. A dataset can be used by accessing [DatasetCatalog](https://github.com/facebookresearch/vissl/tree/main/vissl/data/dataset_catalog.json)
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

## Expected dataset structure for CIFAR-10

The expected format is the exact same format used by torchvision, and the exact format obtained after either:
- expanding the "CIFAR-10 python version" archive available at https://www.cs.toronto.edu/~kriz/cifar.html
- instantiating the `torchvision.datasets.CIFAR10` class with `download=True`

```
cifar-10-batches-py/
    batches.meta
    data_batch_1
    data_batch_2
    data_batch_3
    data_batch_4
    data_batch_5
    readme.html
    test_batch
```

## Expected dataset structure for CIFAR-100

The expected format is the exact same format used by torchvision, and the exact format obtained after either:
- expanding the "CIFAR-100 python version" archive available at https://www.cs.toronto.edu/~kriz/cifar.html
- instantiating the `torchvision.datasets.CIFAR100` class with `download=True`

```
cifar-100-python/
    meta
    test
    train
```

## Expected dataset structure for MNIST

The expected format is the exact same format used by torchvision, and the exact format obtained after
instantiating the `torchvision.datasets.MNIST` class with the flag `download=True`.

```
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
```

## Expected dataset structure for STL10

The expected format is the exact same format used by torchvision, and the exact format obtained after either:
- expanding the `stl10_binary.tar.gz` archive available at https://cs.stanford.edu/~acoates/stl10/
- instantiating the `torchvision.datasets.STL10` class with `download=True`

```
stl10_binary/
    class_names.txt
    fold_indices.txt
    test_X.bin
    test_y.bin
    train_X.bin
    train_y.bin
    unlabeled_X.bin
```

## Expected dataset structure for SVHN

The expected format is the exact same format used by torchvision, and the exact format obtained after either:

- downloading the `train_32x32.mat`, `test_32x32.mat` and `extra_32x32.mat` files available at http://ufldl.stanford.edu/housenumbers/ in the same folder

- instantiating the `torchvision.datasets.SVHN` class with `download=True`

```bash
svhn_folder/
    test_32x32.mat
    train_32x32.mat
```

## Expected dataset structure for the other benchmark datasets

VISSL supports benchmarks inspired by the [VTAB](https://arxiv.org/pdf/1910.04867.pdf) and [CLIP](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf) papers, for which the datasets either:

- Do not directly exist but are transformations of existing dataset (like images extracted from videos)
- Are not in a format directly compatible with the `disk_folder` or the `disk_filelist` format of VISSL
- And are not yet part of [torchvision](https://pytorch.org/vision/stable/datasets.html) datasets

To run these benchmarks, the following data preparation scripts are mandatory:

- `extra_scripts/datasets/create_caltech101_data_files.py`: to transform the [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_clevr_count_data_files.py`: to create a `disk_filelist` dataset from [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) where the goal is to count the number of object in the scene
- `extra_scripts/datasets/create_clevr_dist_data_files.py`: to create a `disk_filelist` dataset from [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) where the goal is to estimate the distance of the closest object in the scene
- `extra_scripts/datasets/create_dsprites_location_data_files.py`: to create a `disk_folder` dataset from [dSprites](https://github.com/deepmind/dsprites-dataset) where the goal is to estimate the x coordinate of the sprite on the scene
- `extra_scripts/datasets/create_dsprites_orientation_data_files.py`: to create a `disk_folder` dataset from [dSprites](https://github.com/deepmind/dsprites-dataset) where the goal is to estimate the orientation of the sprite on the scene
- `extra_scripts/datasets/create_dtd_data_files.py`: to transform the [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_euro_sat_data_files.py`: to transform the [EUROSAT](https://github.com/phelber/eurosat) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_fgvc_aircraft_data_files.py`: to transform the [FGVC Aircrafts](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_food101_data_files.py`: to transform the [FOOD101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_gtsrb_data_files.py`: to transform the [GTSRB](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_kitti_dist_data_files.py`: to create a `disk_folder` dataset from [KITTI](http://www.cvlibs.net/datasets/kitti/) where the goal is to estimate the distance of the closest car, van or truck
- `extra_scripts/datasets/create_oxford_pets_data_files.py`: to transform the [Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_patch_camelyon_data_files.py`: to transform the [PatchCamelyon](https://github.com/basveeling/pcam) dataset to the `disk_folder` format
- `extra_scripts/datasets/create_small_norb_azimuth_data_files.py` to create a `disk_folder` dataset from [Small NORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) where the goal is to find the azimuth or the photographed object
- `extra_scripts/datasets/create_small_norb_elevation_data_files.py` to create a `disk_folder` dataset from [Small NORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) where the goal is to predict the elevation in the image
- `extra_scripts/datasets/create_ucf101_data_files.py`: to create a `disk_folder` image action recognition dataset from the video action recognition dataset [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) by extracting the middle frame

You can read more about how to download these datasets and run these scripts from [here](https://github.com/facebookresearch/vissl/blob/main/extra_scripts/README.md).

After data preparation, the output folders are compatible with the `disk_folder` layout or `disk_filelist` layout.

## Dataset Catalog
It contains a mapping from strings (which are names that identify a dataset,
e.g. "imagenet1k_folder") to a `dict` which contains:
1. mapping of various data splits (train, test, val) to the data source (path on the disk whether a folder path or a filelist)
2. source of the data (`disk_filelist` | `disk_folder` | `torchvision_dataset`)
    - `disk_folder`: this is simply the root folder path to the downloaded data.
    - `disk_filelist`: These are numpy files: (1) file containing images information (2) file containing corresponding labels for images. We provide [scripts](https://github.com/facebookresearch/vissl/blob/main/extra_scripts/README.md) that can be used to prepare these two files for a dataset of choice.
    - `torchvision_dataset`: the root folder path to the torchvision dowloaded dataset. As of now, the supported datasets are: CIFAR-10, CIFAR-100, MNIST and STL-10.

The purpose of having this catalog is to make it easy to choose different datasets,
by just using the strings in the config.

### Creating a custom Dataset catalog `dataset_catalog.json`

Users can edit the template `vissl/configs/config/dataset_catalog.json` file to specify their datasets. The json file can be fully decided by user and can have any number of supported datasets (one or more). User can give the string names to dataset as per their choice.

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
