# VISSL Model Zoo and Benchmarks

VISSL provides reference implementation of a large number of self-supervision approaches and also a suite of benchmark tasks to quickly evaluate the representation quality of models trained with these self-supervised tasks using standard evaluation setup. In this document, we list the collection of self-supervised models and benchmark of these models on a standard task of evaluating a linear classifier on ImageNet-1K. All the models can be downloaded from the provided links.

## Table of Contents
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
   - [DeepClusterV2](#DeepClusterV2)
   - [SwAV](#SwAV)
   - [MoCoV2](#MoCoV2)

## Models

VISSL is 100% compatible with TorchVision ResNet models. You can benchmark these models using VISSL's benchmark suite. See the docs for how to run various benchmarks.

### Supervised
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| Supervised      |    [RN50 - Torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)                             |     ImageNet      | -- | [model]()
| Supervised      |    [RN101 - Torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)                            |     ImageNet      | -- | [model]()
| Supervised      |    [RN50 - Caffe2](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)         |     ImageNet      | -- | [model]()
| Supervised      |    [RN50 - Caffe2](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)         |     Places205     | -- | [model]()
| Supervised      |    [Alexnet BVLC - Caffe2](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models) |     ImageNet      | -- | [model]()
| Supervised      |    RN50 - VISSL - 105 epochs                                                                                                    |     ImageNet      | -- | [model]()

### Semi-weakly and Semi-supervised
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [Semi-supervised](https://arxiv.org/abs/1905.00546) | [RN50](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/blob/master/hubconf.py)        | YFCC100M - ImageNet                | -- | [model]()
| [Semi-weakly supervised](https://arxiv.org/abs/1905.00546) | [RN50](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/blob/master/hubconf.py) | Public Instagram Images - ImageNet | -- | [model]()

### Jigsaw
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    RN50 - 100 permutations                                                                                                      | ImageNet-1K  | -- | [model]()
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    RN50 - 2K permutations                                                                                                       | ImageNet-1K  | -- | [model]()
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    RN50 - 10K permutations                                                                                                      | ImageNet-1K  | -- | [model]()
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    RN50 - 2K permutations                                                                                                       | ImageNet-22K | -- | [model]()
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)       | ImageNet-1K  | -- | [model]()
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)       | ImageNet-22K | -- | [model]()
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)       | YFCC100M     | -- | [model]()
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | ImageNet-1K  | -- | [model]()
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | ImageNet-22K | -- | [model]()
| [Jigsaw](https://arxiv.org/abs/1603.09246)      |    [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | YFCC100M     | -- | [model]()

### Colorization
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [Colorization](https://arxiv.org/abs/1603.08511) | [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | ImageNet-1K  | -- | [model]()
| [Colorization](https://arxiv.org/abs/1603.08511) | [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | ImageNet-22K | -- | [model]()
| [Colorization](https://arxiv.org/abs/1603.08511) | [RN50 - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models)    | YFCC100M     | -- | [model]()
| [Colorization](https://arxiv.org/abs/1603.08511) | [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models) | ImageNet-1K  | -- | [model]()
| [Colorization](https://arxiv.org/abs/1603.08511) | [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models) | ImageNet-22K | -- | [model]()
| [Colorization](https://arxiv.org/abs/1603.08511) | [AlexNet - Goyal'19](https://github.com/facebookresearch/fair_self_supervision_benchmark/blob/master/MODEL_ZOO.md#models) | YFCC100M     | -- | [model]()

### RotNet
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [RotNet](https://arxiv.org/abs/1803.07728) | RN50 - 105 epochs                                                                                               | ImageNet-1K | -- | [model]()
| [RotNet](https://arxiv.org/abs/1803.07728) | RN50 - 105 epochs                                                                                               | ImageNet-22K | -- | [model]()
| [RotNet](https://arxiv.org/abs/1803.07728) | [AlexNet official](https://github.com/gidariss/FeatureLearningRotNet#download-the-already-trained-rotnet-model) | ImageNet-1K | -- | [model]()

### DeepCluster
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [DeepCluster](https://arxiv.org/abs/1807.05520)   |    [AlexNet official](https://github.com/facebookresearch/deepcluster#pre-trained-models)   |   ImageNet-1K      | -- | [model]()

### ClusterFit
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [ClusterFit](https://arxiv.org/abs/1912.03330)    |    RN50 - 105 epochs - 16K clusters from RotNet  |  ImageNet-1K   | -- | [model]()

### NPID
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [NPID](https://arxiv.org/abs/1805.01978)  |    [RN50 official oldies](https://github.com/zhirongw/lemniscate.pytorch#updated-pretrained-model)   |  ImageNet-1K | -- | [model]()
| [NPID](https://arxiv.org/abs/1805.01978)  |    RN50 - 4k negatives - 200 epochs - VISSL                                                          |  ImageNet-1K | -- | [model]()

### NPID++
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [NPID++](https://arxiv.org/abs/1912.01991)      |    RN50 - 32k negatives - 800 epochs - cosine LR       |      ImageNet-1K      | -- | [model]()
| [NPID++](https://arxiv.org/abs/1912.01991)      |    RN50-w2 - 32k negatives - 800 epochs - cosine LR    |      ImageNet-1K      | -- | [model]()

### PIRL
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [PIRL](https://arxiv.org/abs/1912.01991)      |    RN50 - 200 epochs       |      ImageNet-1K      | -- | [model]()
| [PIRL](https://arxiv.org/abs/1912.01991)      |    RN50 - 800 epochs       |      ImageNet-1K      | -- | [model]()

**NOTE:** Please see [projects/PIRL/README.md](https://github.com/facebookresearch/vissl/blob/master/projects/PIRL/README.md) for more PIRL models provided by authors.

### SimCLR
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 100 epochs       |      ImageNet-1K      | -- | [model]()
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 200 epochs       |      ImageNet-1K      | -- | [model]()
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 400 epochs       |      ImageNet-1K      | -- | [model]()
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 800 epochs       |      ImageNet-1K      | -- | [model]()
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50 - 1000 epochs      |      ImageNet-1K      | -- | [model]()
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50-w2 - 100 epochs    |      ImageNet-1K      | -- | [model]()
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50-w2 - 1000 epochs   |      ImageNet-1K      | -- | [model]()
| [SimCLR](https://arxiv.org/abs/2002.05709)      |    RN50-w4 - 1000 epochs   |      ImageNet-1K      | -- | [model]()

### DeepClusterV2
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [DeepClusterV2](https://arxiv.org/abs/2006.09882)  |  [RN50 - 400 epochs - 2x224](https://github.com/facebookresearch/swav#model-zoo)       |  ImageNet-1K  | -- | [model]()
| [DeepClusterV2](https://arxiv.org/abs/2006.09882)  |  [RN50 - 400 epochs - 2x160+4x96](https://github.com/facebookresearch/swav#model-zoo)  |  ImageNet-1K  | -- | [model]()

### SwAV
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 100 epochs - 2x224+6x96 - 4096 batch-size    |    ImageNet-1K      | -- | [model]()
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 200 epochs - 2x224+6x96 - 4096 batch-size    |    ImageNet-1K      | -- | [model]()
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 400 epochs - 2x224+6x96 - 4096 batch-size    |    ImageNet-1K      | -- | [model]()
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 800 epochs - 2x224+6x96 - 4096 batch-size    |    ImageNet-1K      | -- | [model]()
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 200 epochs - 2x224+6x96 - 256 batch-size     |    ImageNet-1K      | -- | [model]()
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 400 epochs - 2x224+6x96 - 256 batch-size     |    ImageNet-1K      | -- | [model]()
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50 - 400 epochs - 2x224 - 4096 batch-size         |    ImageNet-1K      | -- | [model]()
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50-w2 - 400 epochs - 2x224+6x96 - 4096 batch-size |    ImageNet-1K      | -- | [model]()
| [SwAV](https://arxiv.org/abs/2006.09882)   |    RN50-w4 - 400 epochs - 2x224+6x96 - 2560 batch-size |    ImageNet-1K      | -- | [model]()

**NOTE:** Please see [projects/SwAV/README.md](https://github.com/facebookresearch/vissl/blob/master/projects/SwAV/README.md) for more SwAV models provided by authors.

### MoCoV2
| Method | Model | PreTrain dataset | ImageNet top-1 acc. | URL |
| ------ | ----- | ---------------- | ------------------- | --- |
| [MoCo-v2](https://arxiv.org/abs/2003.04297)   |    RN50 - 200 epochs - 256 batch-size         |    ImageNet-1K      | -- | [model]()
