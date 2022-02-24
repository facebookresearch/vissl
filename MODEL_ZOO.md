# VISSL Model Zoo and Benchmarks

VISSL provides reference implementation of a large number of self-supervision approaches and also a suite of benchmark tasks to quickly evaluate the representation quality of models trained with these self-supervised tasks using standard evaluation setup. In this document, we list the collection of self-supervised models and benchmark of these models on a standard task of evaluating a linear classifier on ImageNet-1K. All the models can be downloaded from the provided links.

## Table of Contents
- [Torchvision and VISSL](#torchvision-and-vissl)
   - [Converting VISSL to Torchvision](#converting-vissl-to-torchvision)
   - [Converting Torchvision to VISSL](#converting-torchvision-to-vissl)
- [Models](#models)
   - [Supervised](#supervised)
   - [Semi-weakly and Semi-supervised](#Semi-weakly-and-Semi-supervised)
   - [Jigsaw](#jigsaw)
   - [Colorization](#Colorization)
   - [RotNet](#RotNet)
   - [DeepCluster](#DeepCluster)
   - [ClusterFit](#ClusterFit)
   - [NPID](#NPID)
   - [NPID++](#NPID++)
   - [PIRL](#PIRL)
   - [SimCLR](#SimCLR)
   - [SimCLRv2](#SimCLRv2)
   - [BYOL](#BYOL)
   - [DeepClusterV2](#DeepClusterV2)
   - [SwAV](#SwAV)
   - [SEER](#SEER)
   - [MoCoV2](#MoCoV2)
   - [Barlow Twins](#BarlowTwins)
   - [DINO](#DINO)

## Torchvision and VISSL

VISSL is 100% compatible with TorchVision ResNet models. It's easy to use torchvision models in VISSL and to use VISSL models in torchvision.

### Converting VISSL to Torchvision

All the ResNe(X)t models in VISSL can be converted to Torchvision weights. This involves simply removing the `_features_blocks.` prefix from all the weights. VISSL provides a convenience script for this:

```bash
python extra_scripts/convert_vissl_to_torchvision.py \
    --model_url_or_file <input_model>.pth  \
    --output_dir /path/to/output/dir/ \
    --output_name <my_converted_model>.torch
```

### Converting Torchvision to VISSL

All the ResNe(X)t models in Torchvision can be directly loaded in VISSL. This involves simply setting the `REMOVE_PREFIX`, `APPEND_PREFIX` options in the config file following the [instructions here](https://github.com/facebookresearch/vissl/blob/main/vissl/config/defaults.yaml#L418-L435). Also, see the example below for how torchvision models are loaded.


## Models

VISSL is 100% compatible with TorchVision ResNet models. You can benchmark these models using VISSL's benchmark suite. See the docs for how to run various benchmarks.

### Supervised

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_supervised.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| Supervised     |    [RN50 - Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)                             |     ImageNet      | 76.1 | [model](https://download.pytorch.org/models/resnet50-19c8e357.pth)
| Supervised     |    [RN101 - Torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)                            |     ImageNet      | 77.21 | [model](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
| Supervised     |    [RN50 - Caffe2](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)         |     ImageNet      | 75.88 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_supervised_in1k_caffe2.torch)
| Supervised     |    [RN50 - Caffe2](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)         |     Places205     | 58.49 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_supervised_places205_caffe2.torch)
| Supervised     |    [Alexnet BVLC - Caffe2](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models) |     ImageNet      | 49.54 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_caffenet_bvlc_in1k_supervised.torch)
| Supervised     |    RN50 - VISSL - 105 epochs                                                                                                    |     ImageNet      | 75.45 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/sup_rn50_in1k_ep105_supervised_8gpu_resnet_17_07_20.733dbdee/model_final_checkpoint_phase208.torch)
| Supervised     |    ViT/B16 - 90 epochs (*) |     ImageNet-22K     | 83.38 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/baselines/vit_b16_p16_in22k_ep90_supervised.torch)
| Supervised     |    RegNetY-64Gf - BGR input | ImageNet | 80.55 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/sup_regnet64/imnetlabels_regnety64gf_3_vissl_converted_bgr.torch)
| Supervised     |    RegNetY-128Gf - BGR input | ImageNet |  80.57 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/sup_regnet128_in1k/imnetlabels_regnety128gf_vissl_converted_bgr.torch)

_(*) This specific checkpoint for ViT/B16 requires the following options to be added in command line to be loaded by VISSL: `config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk.base_model. config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=classy_state_dict`_

### Semi-weakly and Semi-supervised

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_supervised.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [Semi-supervised](https://arxiv.org/abs/1905.00546) | [RN50](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/blob/master/hubconf.py)        | YFCC100M - ImageNet        | 79.2 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_semi_sup_08389792.torch)
| [Semi-weakly supervised](https://arxiv.org/abs/1905.00546) | [RN50](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/blob/master/hubconf.py) | Public Instagram Images - ImageNet | 81.06 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_semi_weakly_sup_16a12f1b.torch)

### Jigsaw

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_jigsaw.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    RN50 - 100 permutations                                                                                                | ImageNet-1K  | 48.57 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.db174a43/model_final_checkpoint_phase104.torch)
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    RN50 - 2K permutations                                                                                                 | ImageNet-1K  | 46.73 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.cccee144/model_final_checkpoint_phase104.torch)
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    RN50 - 10K permutations                                                                                                | ImageNet-1K  | 48.11 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_20_07_20.3d706467/model_final_checkpoint_phase104.torch)
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    RN50 - 2K permutations                                                                                                       | ImageNet-22K | 44.84 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_jigsaw_rn50_perm2k_in22k_8gpu_ep105.torch)
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)       | ImageNet-1K  | 46.58 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_in1k_goyal19.torch)
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)       | ImageNet-22K | 53.09 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_in22k_goyal19.torch)
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)       | YFCC100M     | 51.37 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_yfcc100m_goyal19.torch)
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | ImageNet-1K  | 34.82 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_alexnet_jigsaw_in1k_pretext.torch)
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | ImageNet-22K | 37.5 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_alexnet_jigsaw_in22k_pretext.torch)
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | YFCC100M     | 37.01 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_alexnet_jigsaw_yfcc100m_pretext.torch)

### Colorization

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_colorization.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [Colorization](https://arxiv.org/abs/1603.08511) | [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)  | ImageNet-1K  | 40.11 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_colorization_in1k_goyal19.torch)
| [Colorization](https://arxiv.org/abs/1603.08511) | [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | ImageNet-22K | 49.24 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_colorization_in22k_goyal19.torch)
| [Colorization](https://arxiv.org/abs/1603.08511) | [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | YFCC100M     | 47.46 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_colorization_yfcc100m_goyal19.torch)
| [Colorization](https://arxiv.org/abs/1603.08511) | [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models) | ImageNet-1K  | 30.39 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_alexnet_colorization_in1k_pretext.torch)
| [Colorization](https://arxiv.org/abs/1603.08511) | [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models) | ImageNet-22K | 36.83 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_alexnet_colorization_in22k_pretext.torch)
| [Colorization](https://arxiv.org/abs/1603.08511) | [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models) | YFCC100M     | 34.19 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_alexnet_colorization_yfcc100m_pretext.torch)

### RotNet

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_rotnet_deepcluster_clusterfit.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [RotNet](https://arxiv.org/abs/1803.07728) | [AlexNet official](https://github.com/gidariss/FeatureLearningRotNet#download-the-already-trained-rotnet-model) | ImageNet-1K  | 39.51 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_alexnet_model_net_epoch50.torch)
| [RotNet](https://arxiv.org/abs/1803.07728) | RN50 - 105 epochs                                                                                               | ImageNet-1K  | 48.2 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch)
| [RotNet](https://arxiv.org/abs/1803.07728) | RN50 - 105 epochs                                                                                               | ImageNet-22K | 54.89 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_in22k_ep105.torch)

### DeepCluster

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_rotnet_deepcluster_clusterfit.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [DeepCluster](https://arxiv.org/abs/1807.05520)   |    [AlexNet official](https://github.com/facebookresearch/deepcluster#pre-trained-models)   |   ImageNet-1K      | 37.88 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/deepcluster_alexnet_checkpoint.torch)

### ClusterFit

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_rotnet_deepcluster_clusterfit.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [ClusterFit](https://arxiv.org/abs/1912.03330)    |    RN50 - 105 epochs - 16K clusters from RotNet  |  ImageNet-1K   | 53.63 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_16kclusters_in1k_ep105.torch)

### NPID

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_npid_pirl.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [NPID](https://arxiv.org/abs/1805.01978)  |    [RN50 official oldies](https://github.com/zhirongw/lemniscate.pytorch#updated-pretrained-model)   |  ImageNet-1K | 54.99 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_npid_lemniscate_neg4k_stepLR_8gpu.torch)
| [NPID](https://arxiv.org/abs/1805.01978)  |    RN50 - 4k negatives - 200 epochs - VISSL                                                          |  ImageNet-1K | 52.73 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_1node_200ep_4kneg_npid_8gpu_resnet_23_07_20.9eb36512/model_final_checkpoint_phase199.torch)

### NPID++

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_npid_pirl.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [NPID++](https://arxiv.org/abs/1912.01991)      |    RN50 - 32k negatives - 800 epochs - cosine LR       |      ImageNet-1K      | 56.68 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_pp/4node_800ep_32kneg_cosine_resnet_23_07_20.75432662/model_final_checkpoint_phase799.torch)
| [NPID++](https://arxiv.org/abs/1912.01991)      |    RN50-w2 - 32k negatives - 800 epochs - cosine LR    |      ImageNet-1K      | 62.73 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_4node_800ep_32kneg_cosine_rn50w2_npid++_4nodes_resnet_27_07_20.b7f4016c/model_final_checkpoint_phase799.torch)

### PIRL

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_npid_pirl.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [PIRL](https://arxiv.org/abs/1912.01991)      |    RN50 - 200 epochs       |      ImageNet-1K      | 62.55 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_200ep_pirl_jigsaw_4node_resnet_22_07_20.ffd17b75/model_final_checkpoint_phase199.torch)
| [PIRL](https://arxiv.org/abs/1912.01991)      |    RN50 - 800 epochs       |      ImageNet-1K      | 64.29 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch)

**NOTE:** Please see [projects/PIRL/README.md](https://github.com/facebookresearch/vissl/blob/main/projects/PIRL/README.md) for more PIRL models provided by authors.

### SimCLR

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_simclr.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 100 epochs       |      ImageNet-1K      | 64.4  | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_100ep_simclr_8node_resnet_16_07_20.8edb093e/model_final_checkpoint_phase99.torch)
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 200 epochs       |      ImageNet-1K      | 66.61 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_200ep_simclr_8node_resnet_16_07_20.a816c0ef/model_final_checkpoint_phase199.torch)
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 400 epochs       |      ImageNet-1K      | 67.71 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_400ep_simclr_8node_resnet_16_07_20.36b338ef/model_final_checkpoint_phase399.torch)
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 800 epochs       |      ImageNet-1K      | 69.68 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch)
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 1000 epochs      |      ImageNet-1K      | 68.8  | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch)
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50-w2 - 100 epochs    |      ImageNet-1K      | 69.82 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50w2_100ep_simclr_8node_resnet_16_07_20.05b37ec3/model_final_checkpoint_phase99.torch)
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50-w2 - 1000 epochs   |      ImageNet-1K      | 73.84 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50w2_1000ep_simclr_8node_resnet_16_07_20.e1e3bbf0/model_final_checkpoint_phase999.torch)
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50-w4 - 1000 epochs   |      ImageNet-1K      | 71.61 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50w4_1000ep_bs32_16node_simclr_8node_resnet_28_07_20.9e20b0ae/model_final_checkpoint_phase999.torch)
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN101 - 100 epochs      |      ImageNet-1K      | 62.76 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_100ep_simclr_8node_resnet_16_07_20.1ff6cb4b/model_final_checkpoint_phase99.torch)
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN101 - 1000 epochs     |      ImageNet-1K      | 71.56 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch)

### SimCLRv2

The following models are converted from the TensorFlow format of the [official repository](https://github.com/google-research/simclr) to VISSL compatible format.

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [SimCLRv2](https://arxiv.org/abs/2006.10029) | [RN152-w3-sk SimCLRv2 repository](https://github.com/google-research/simclr) | ImageNet-1K | 80.0 |  [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/baselines/converted_simclr_v2_r152_3x_sk1_ema.torch) |

### BYOL

The following models are converted from the TensorFlow format of the [official repository](https://github.com/deepmind/deepmind-research/tree/master/byol) to VISSL compatible format.

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [BYOL](https://arxiv.org/abs/2006.07733) | [RN200-w2 BYOL repository](https://github.com/deepmind/deepmind-research/tree/master/byol) (*) | ImageNet-1K | 78.34 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/baselines/converted_byol_pretrain_res200w2.torch) |

_(*) This specific checkpoint requires the following command line options to be provided to VISSL to be correctly loaded by VISSL: `config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk.base_model._feature_blocks. config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=''`_

### DeepClusterV2

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_deepclusterv2_swav.json).

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [DeepClusterV2](https://arxiv.org/abs/2006.09882)  |  [RN50 - 400 epochs - 2x224](https://github.com/facebookresearch/swav#model-zoo)       |  ImageNet-1K  | 70.01 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_400ep_2x224_pretrain.pth.tar)
| [DeepClusterV2](https://arxiv.org/abs/2006.09882)  |  [RN50 - 400 epochs - 2x160+4x96](https://github.com/facebookresearch/swav#model-zoo)  |  ImageNet-1K  | 74.32 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_400ep_pretrain.pth.tar)
| [DeepClusterV2](https://arxiv.org/abs/2006.09882)  |  [RN50 - 800 epochs - 2x224+6x96](https://github.com/facebookresearch/swav#model-zoo)  |  ImageNet-1K  | 75.18 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_800ep_pretrain.pth.tar)

### SwAV

To reproduce the numbers below, the experiment configuration is provided in json format for each model [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/model_zoo/benchmark_in1k_linear_deepclusterv2_swav.json).

There is some standard deviation in linear results if we run the same eval several times and pre-train a SwAV model several times. The evals reported below are for 1 run.

| Method | Model | PreTrain dataset | ImageNet top-1 linear acc. | URL |
| ------ | ----- | ---------------- |:--------------------------:| --- |
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 100 epochs - 2x224+6x96 - 4096 batch-size    |    ImageNet-1K      | 71.99 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_100ep_swav_8node_resnet_27_07_20.7e6fc6bf/model_final_checkpoint_phase99.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 200 epochs - 2x224+6x96 - 4096 batch-size    |    ImageNet-1K      | 73.85 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_200ep_swav_8node_resnet_27_07_20.bd595bb0/model_final_checkpoint_phase199.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 400 epochs - 2x224+6x96 - 4096 batch-size    |    ImageNet-1K      | 74.81 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_400ep_swav_8node_resnet_27_07_20.a5990fc9/model_final_checkpoint_phase399.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 800 epochs - 2x224+6x96 - 4096 batch-size    |    ImageNet-1K      | 74.92 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/model_final_checkpoint_phase799.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 200 epochs - 2x224+6x96 - 256 batch-size     |    ImageNet-1K      | 73.07 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_4gpu_bs64_200ep_2x224_6x96_queue_swav_8node_resnet_28_07_20.a8f2c735/model_final_checkpoint_phase199.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 400 epochs - 2x224+6x96 - 256 batch-size     |    ImageNet-1K      | 74.3  | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_4gpu_bs64_400ep_2x224_6x96_queue_swav_8node_resnet_28_07_20.5e967ca0/model_final_checkpoint_phase399.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 400 epochs - 2x224 - 4096 batch-size         |    ImageNet-1K      | 69.53 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_8node_2x224_rn50_in1k_swav_8node_resnet_30_07_20.c8fd7169/model_final_checkpoint_phase399.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50-w2 - 400 epochs - 2x224+6x96 - 4096 batch-size |    ImageNet-1K      | 77.01 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_rn50w2_in1k_bs32_16node_400ep_swav_8node_resnet_30_07_20.93563e51/model_final_checkpoint_phase399.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50-w4 - 400 epochs - 2x224+6x96 - 2560 batch-size |    ImageNet-1K      | 77.03 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_rn50w4_in1k_bs40_8node_400ep_swav_8node_resnet_30_07_20.1736135b/model_final_checkpoint_phase399.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50-w5 - 300 epochs - 2x224+6x96 - 2560 batch-size (*)   |    ImageNet-1K      | 78.5  | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_rn50w5/swav_RN50w5_400ep_pretrain.pth.tar)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RegNetY-16Gf - 800 epochs - 2x224+6x96 - 4096 batch-size  |    ImageNet-1K      | 76.15 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_regnet16_in1k/swav_in1k_regnet16gf_model_final_checkpoint_phase799.torch)
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RegNetY-128Gf - 400 epochs - 2x224+6x96 - 4096 batch-size |    ImageNet-1K      | 78.36 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_regnet128_in1k/model_phase230.torch)

**NOTE:** Please see [projects/SwAV/README.md](https://github.com/facebookresearch/vissl/blob/main/projects/SwAV/README.md) for more SwAV models provided by authors.

_(*) This specific RN50-w5 checkpoint requires the following options to be added to be loaded by VISSL: `config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk.base_model._feature_blocks. config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME='' config.MODEL.WEIGHTS_INIT.REMOVE_PREFIX=module.`_

### SEER

| Method | Model | PreTrain dataset | ImageNet top-1 linear acc. | ImageNet top-1 fine-tuned acc. | URL |
| ------ | ----- | ---------------- |:-------------------:|:---:| --- |
| [SEER]() | RegNetY-32Gf  | IG-1B public images, non EU | 74.03 (res5) | 83.4 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet32d/seer_regnet32gf_model_iteration244000.torch) |
| [SEER]() | RegNetY-64Gf  | IG-1B public images, non EU | 75.25 (res5avg) | 84.0 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet64/seer_regnet64gf_model_final_checkpoint_phase0.torch) |
| [SEER]() | RegNetY-128Gf | IG-1B public images, non EU | 75.96 (res5avg) | 84.5 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_regnet128Gf_cnstant_bs32_node16_sinkhorn10_proto16k_syncBN64_warmup8k/model_final_checkpoint_phase0.torch) |
| [SEER]() | RegNetY-256Gf | IG-1B public images, non EU | 77.51 (res5avg) | 85.2 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_ig1b_cosine_rg256gf_noBNhead_wd1e5_fairstore_bs16_node64_sinkhorn10_proto16k_apex_syncBN64_warmup8k/model_final_checkpoint_phase0.torch) |
| [SEER]() | RegNet10B   | IG-1B public images, non EU | 79.8 (res4) | 85.8 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/seer_regnet10B/model_iteration124500_conso.torch) |

**NOTE:** Please see [projects/SEER/README.md](/projects/SEER/README.md) for more SwAV models provided by authors.

### MoCoV2

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [MoCo-v2](https://arxiv.org/abs/2003.04297)   |    RN50 - 200 epochs - 256 batch-size         |    ImageNet-1K      | 66.4 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/moco_v2_1node_lr.03_step_b32_zero_init/model_final_checkpoint_phase199.torch)

### MoCoV3

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [MoCo-v3](https://arxiv.org/abs/2104.02057)  |    ViT-B/16 - 300 epochs        |    ImageNet-1K      | 75.79 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/swav_rn50w5/mocov3-vit-b-300ep.pth.tar)

### BarlowTwins

| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [Barlow Twins](https://arxiv.org/abs/2103.03230)   |    RN50 - 300 epochs - 2048 batch-size         |    ImageNet-1K      | 70.75 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_300ep_resnet50.torch)
| [Barlow Twins](https://arxiv.org/abs/2103.03230)   |    RN50 - 1000 epochs - 2048 batch-size         |    ImageNet-1K      | 71.80 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch)

### DINO

The ViT-small model is obtained with [this config](https://github.com/facebookresearch/vissl/blob/main/configs/config/pretrain/dino/dino_16gpus_deits16.yaml).

| Method | Model | PreTrain dataset | ImageNet k-NN acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [DINO](https://arxiv.org/abs/2104.14294)   |    ViT-S/16 - 300 epochs - 1024 batch-size         |    ImageNet-1K      | 73.4 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/dino_300ep_deitsmall16/model_final_checkpoint_phase299.torch)
| [DINO](https://arxiv.org/abs/2104.14294)   |    XCiT-S/16 - 300 epochs - 1024 batch-size         |    ImageNet-1K      | 74.8 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/dino_300ep_xcitsmall16/model_phase250.torch)
