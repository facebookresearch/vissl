<img src=".github/logo/Logo_Color_Light_BG.png" width="900"/>

[![CircleCI](https://circleci.com/gh/facebookresearch/vissl.svg?style=svg&circle-token=a0c1dfc4edf0ebe523700b95a563e74eb4d4e5a8)](https://circleci.com/gh/facebookresearch/vissl)[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/vissl/blob/main/.github/CONTRIBUTING.md)

## What's New
Below we share, in reverse chronological order, the updates and new releases in VISSL. All VISSL releases are available [here](https://github.com/facebookresearch/vissl/releases).
- **[Oct 2021]**: [Vissl Release 0.1.6](https://github.com/facebookresearch/vissl/releases/tag/v0.1.6) We have released a new version of VISSL! Please see our [release notes](https://github.com/facebookresearch/vissl/releases/tag/v0.1.6) for more information.
- **[Oct 2021]**: [AugLy](https://github.com/facebookresearch/AugLy) data augmentations support introduced in this [commit](https://github.com/facebookresearch/vissl/commit/dd9971a0c1fb17c96fd4994d73ebbda6f82c8f03).
- **[Oct 2021]**: [XCiT: Cross-Covariance Image Transformers](https://arxiv.org/pdf/2106.09681.pdf) code released in this [commit](https://github.com/facebookresearch/vissl/commit/056e695a84ed1e1c63320ddeecb577a9e03cde63).
- **[Sept 2021]**: VISSL master branch renamed to main in this [PR](https://github.com/facebookresearch/vissl/pull/425) in VISSL.
- **[August 2021]**: Instance Retrieval benchmark implemented and [available](https://github.com/facebookresearch/vissl/pull/394) in VISSL.
- **[July 2021]**: [Fully Sharded Data Parallel](https://github.com/facebookresearch/fairscale/pull/413) integrated in VISSL and announced in [blog](https://engineering.fb.com/2021/07/15/open-source/fsdp/).
- **[May 2021]**: [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.14294.pdf) code released.
- **[May 2021]**: VISSL relicensed under [MIT License](LICENSE).
- **[May 2021]**: [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) code released.
- **[April 2021]**: [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/pdf/2103.10697.pdf) code released.
- **[March 2021]**: Added most benchmark datasets used in [VTAB](https://arxiv.org/pdf/1910.04867.pdf) and [CLIP](https://arxiv.org/abs/2103.00020) benchmark tasks.
- **[February 2021]**: Added [Vision Transformers](https://arxiv.org/pdf/2010.11929v1.pdf) (ViT) backbone and [training self-supervision with ViT](https://github.com/facebookresearch/vissl/tree/main/configs/config/pretrain/vision_transformer).
- **[January 2021]**: VISSL v0.1.5 released.

## Introduction
VISSL is a computer **VI**sion library for state-of-the-art **S**elf-**S**upervised **L**earning research with [PyTorch](https://pytorch.org). VISSL aims to accelerate research cycle in self-supervised learning: from designing a new self-supervised task to evaluating the learned representations. Key features include:

- **Reproducible implementation of SOTA in Self-Supervision**: All existing SOTA in Self-Supervision are implemented - [SwAV](https://arxiv.org/abs/2006.09882), [SimCLR](https://arxiv.org/abs/2002.05709), [MoCo(v2)](https://arxiv.org/abs/1911.05722), [PIRL](https://arxiv.org/abs/1912.01991), [NPID](https://arxiv.org/pdf/1805.01978.pdf), [NPID++](https://arxiv.org/abs/1912.01991), [DeepClusterV2](https://arxiv.org/abs/2006.09882), [ClusterFit](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yan_ClusterFit_Improving_Generalization_of_Visual_Representations_CVPR_2020_paper.pdf), [RotNet](https://arxiv.org/abs/1803.07728), [Jigsaw](https://arxiv.org/abs/1603.09246). Also supports supervised trainings.

- **Benchmark suite**: Variety of benchmarks tasks including [linear image classification (places205, imagenet1k, voc07, food, CLEVR, dsprites, UCF101, stanford cars and many more)](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/linear_image_classification), [full finetuning](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/fulltune), [semi-supervised benchmark](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/semi_supervised), [nearest neighbor benchmark](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/nearest_neighbor), [object detection (Pascal VOC and COCO)](https://github.com/facebookresearch/vissl/tree/main/configs/config/benchmark/object_detection).

- **Ease of Usability**: easy to use using yaml configuration system based on [Hydra](https://github.com/facebookresearch/hydra).

- **Modular**: Easy to design new tasks and reuse the existing components from other tasks (objective functions, model trunk and heads, data transforms, etc.). The modular components are simple *drop-in replacements* in yaml config files.

- **Scalability**: Easy to train model on 1-gpu, multi-gpu and multi-node. Several components for large scale trainings provided as simple config file plugs: [Activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html), [ZeRO](https://arxiv.org/abs/1910.02054), [FP16](https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use), [LARC](https://arxiv.org/abs/1708.03888), Stateful data sampler, data class to handle invalid images, large model backbones like [RegNets](https://arxiv.org/abs/2003.13678), etc.

- **Model Zoo**: Over *60 pre-trained self-supervised model* weights.

## Installation

See [`INSTALL.md`](https://github.com/facebookresearch/vissl/blob/main/INSTALL.md).

## Getting Started

Install VISSL by following the [installation instructions](https://github.com/facebookresearch/vissl/blob/main/INSTALL.md).
After installation, please see [Getting Started with VISSL](https://github.com/facebookresearch/vissl/blob/main/GETTING_STARTED.md) and the [Colab Notebook](https://colab.research.google.com/github/facebookresearch/vissl/blob/v0.1.6/tutorials/Understanding_VISSL_Training_and_YAML_Config_V0_1_6.ipynb) to learn about basic usage.

## Documentation

Learn more about VISSL at our [documentation](https://vissl.readthedocs.io). And see the [projects/](projects/) for some projects built on top of VISSL.

## Tutorials

Get started with VISSL by trying one of the **Colab tutorial notebooks**.

- [Train SimCLR on 1-gpu](https://colab.research.google.com/github/facebookresearch/vissl/blob/v0.1.6/tutorials/Train_SimCLR_on_1_gpu_V0_1_6.ipynb)
- [Extracting Features from a pretrained model](https://colab.research.google.com/github/facebookresearch/vissl/blob/v0.1.6/tutorials/Using_a_pretrained_model_for_inference_V0_1_6.ipynb)
- [Benchmark task: Full finetuning on ImageNet-1K](https://colab.research.google.com/github/facebookresearch/vissl/blob/v0.1.6/tutorials/Benchmark_Full_Finetuning_on_ImageNet_1K_V0_1_6.ipynb)
- [Benchmark task: Linear image classification on ImageNet-1K](https://colab.research.google.com/github/facebookresearch/vissl/blob/v0.1.6/tutorials/Benchmark_Linear_Image_Classification_on_ImageNet_1K_V0_1_6.ipynb)
- [Large scale training (fp16, LARC, ZeRO)](https://colab.research.google.com/github/facebookresearch/vissl/blob/v0.1.6/tutorials/Large_Scale_Training_V0_1_6.ipynb)
- [Using a pre-trained model in inference mode](https://colab.research.google.com/github/facebookresearch/vissl/blob/v0.1.6/tutorials/Using_a_pretrained_model_for_inference_V0_1_6.ipynb)


## Model Zoo and Baselines
We provide a large set of baseline results and trained models available for download in the [VISSL Model Zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md).

## Contributors

VISSL is written and maintained by the Facebook AI Research.

## Development

We welcome new contributions to VISSL and we will be actively maintaining this library! Please refer to [`CONTRIBUTING.md`](./.github/CONTRIBUTING.md) for full instructions on how to run the code, tests and linter, and submit your pull requests.

## License

VISSL is released under [MIT license](LICENSE).

## Citing VISSL

If you find VISSL useful in your research or wish to refer to the baseline results published in the [Model Zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{goyal2021vissl,
  author =       {Priya Goyal and Quentin Duval and Jeremy Reizenstein and Matthew Leavitt and Min Xu and
                  Benjamin Lefaudeux and Mannat Singh and Vinicius Reis and Mathilde Caron and Piotr Bojanowski and
                  Armand Joulin and Ishan Misra},
  title =        {VISSL},
  howpublished = {\url{https://github.com/facebookresearch/vissl}},
  year =         {2021}
}
```
