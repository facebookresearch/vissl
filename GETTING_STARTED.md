# Quick Start with VISSL

We provide a quick overview for training SimCLR self-supervised model on 1-gpu with VISSL.

## Install VISSL
For installation, please refer to [`INSTALL.md`](INSTALL.md).


## ImageNet-1K dataset
We assume the downloaded data to look like:

```
imagenet_full_size
|_ train
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
|_ val
|  |_ <n0......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-N-name>.JPEG
|  |_ ...
|  |_ <n1......>
|  |  |_<im-1-name>.JPEG
|  |  |_...
|  |  |_<im-M-name>.JPEG
|  |  |_...
|  |  |_...
```

## Running SimCLR Pre-training on 1-gpu on ImageNet1K

### If VISSL is built from source
We provide a config to train model using the pretext SimCLR task on the ResNet50 model.
Change the `DATA.TRAIN.DATA_PATHS` path to the ImageNet train dataset folder path.

```bash
cd $HOME/vissl
python3 tools/run_distributed_engines.py \
    hydra.verbose=true \
    config.DATA.TRAIN.DATASET_NAMES=[imagenet1k_folder] \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATA_PATHS=["/path/to/my/imagenet/folder/train"] \
    config=test/integration_test/quick_simclr_imagefolder \
    config.CHECKPOINT.DIR="./checkpoints" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true
```

### If using pre-built conda/pip VISSL packages

Users need to set the dataset and obtain the builtin tool for training. Follow the steps:

#### Step1: Setup ImageNet1K dataset
If you installed pre-built VISSL packages, we will set the ImageNet1K dataset following our [data documentation](https://vissl.readthedocs.io/en/latest/vissl_modules/data.html) and [tutorial](https://colab.research.google.com/drive/1CCuZ50BN99JcOB6VEPytVi_i2tSMd7A3#scrollTo=KPGCiTsXZeW3). NOTE that we need to register
the dataset with VISSL.

In your python interpretor:
```python
>>> json_data = {
        "imagenet1k_folder": {
            "train": ["<img_path>", "<lbl_path>"],
            "val": ["<img_path>", "<lbl_path>"]
        }
    }
>>> from vissl.utils.io import save_file
>>> save_file(json_data, "/tmp/configs/config/dataset_catalog.json", append_to_json=False)
>>> from vissl.data.dataset_catalog import VisslDatasetCatalog
>>> print(VisslDatasetCatalog.list())
['imagenet1k_folder']
>>> print(VisslDatasetCatalog.get("imagenet1k_folder"))
{'train': ['<img_path>', '<lbl_path>'], 'val': ['<img_path>', '<lbl_path>']}
```

#### Step2: Get the builtin tool and yaml config file
We will use the pre-built VISSL tool for training [run_distributed_engines.py](https://github.com/facebookresearch/vissl/blob/stable/tools/run_distributed_engines.py) and the config file. Run

```bash
cd /tmp/ && mkdir -p /tmp/configs/config
wget -q -O configs/__init__.py https://dl.fbaipublicfiles.com/vissl/tutorials/configs/__init__.py
wget -q -O configs/config/quick_1gpu_resnet50_simclr.yaml https://dl.fbaipublicfiles.com/vissl/tutorials/configs/quick_1gpu_resnet50_simclr.yaml
wget -q  https://dl.fbaipublicfiles.com/vissl/tutorials/run_distributed_engines.py
```

#### Step3: Train
```bash
cd /tmp/
python3 run_distributed_engines.py \
    hydra.verbose=true \
    config.DATA.TRAIN.DATASET_NAMES=[imagenet1k_folder] \
    config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
    config.DATA.TRAIN.DATA_PATHS=["/path/to/my/imagenet/folder/train"] \
    config=quick_1gpu_resnet50_simclr \
    config.CHECKPOINT.DIR="./checkpoints" \
    config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true
```

Explore **all the parameters and settings VISSL supports** in [VISSL defaults.yaml file](https://github.com/facebookresearch/vissl/blob/master/vissl/config/defaults.yaml)
