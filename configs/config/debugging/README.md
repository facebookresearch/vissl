## Debugging configurations

This folder is meant for configurations to quickly iterate on ideas and debugging algorithms that are under development.

To do so, these configurations are based on smaller datasets which have proved useful to help reproducing algorithms, such as Imagenette 160px (available on the Fast AI Github at https://github.com/fastai/imagenette).

For instance, on the Github repo of BYOL (https://github.com/deepmind/deepmind-research/tree/master/byol), the "Setup for fast iteration" is based on Imagenette.

Some example of configuration are provided here as reference:

- `debugging/benchmark/linear_image_classification/eval_resnet_8gpu_transfer_imagenette_160` for quickly verifying if representations are useful on a small classification dataset 
- `debugging/pretrain/simclr/simclr_1node_resnet_imagenette_160` as an example of how to use the dataset for SSL algorithm

We encourage enriching this folder with similar configurations for other SSL algorithms under development.
