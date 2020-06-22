# Using SSL framework

We provide a brief tutorial for running various evaluations using various tasks (benchmark/legacy) on various datasets.

- For installation, please refer to [`INSTALL.md`](INSTALL.md).

## Preparing Data input files

Below are the example commands to prepare input data files for various datasets.


### Preparing ImageNet-1K data files
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

Run the following commands to create the data input files:
```bash
mkdir -p $HOME/ssl_scaling/datasets/imagenet1k/

cd $HOME/ssl_scaling
python extra_scripts/create_imagenet_data_files.py \
    --data_source_dir /datasets01_101/imagenet_full_size/061417/ \
    --output_dir $HOME/ssl_scaling/datasets/imagenet1k/
```

## Running Pre-training

We provide a config to train model using the pretext SimCLR task on the ResNet50 model.
Change the `TRAIN.DATA_PATHS` path in the config to where the imagenet handles are saved from the above script.

```bash
python tools/distributed_train.py --node_id 0 \
    --config_file configs/simple_clr/pretext_rn50_2gpu_simpleCLR_imagenet.yaml
```
