<img src=".github/logo/Logo_Color_Light_BG.png" width="900"/>

[![CircleCI](https://circleci.com/gh/facebookresearch/vissl.svg?style=svg&circle-token=a0c1dfc4edf0ebe523700b95a563e74eb4d4e5a8)](https://circleci.com/gh/facebookresearch/vissl)[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/facebookresearch/vissl/blob/master/.github/CONTRIBUTING.md)

## What's New
Below we share, in reverse chronological order, the updates and new releases in VISSL. All VISSL releases are available [here](https://github.com/facebookresearch/vissl/releases).

- **[May 2021]**: [DINO: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.14294.pdf) code released.
- **[May 2021]**: VISSL relicensed under [MIT License](LICENSE).
- **[May 2021]**: [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) code released.
- **[April 2021]**: [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/pdf/2103.10697.pdf) code released.
- **[March 2021]**: Added most datasets used in [VTAB](https://arxiv.org/pdf/1910.04867.pdf) and [CLIP](https://arxiv.org/abs/2103.00020) benchmark tasks.
- **[February 2021]**: Added [Vision Transformers](https://arxiv.org/pdf/2010.11929v1.pdf) (ViT) backbone and [training self-supervision with ViT](https://github.com/facebookresearch/vissl/tree/master/configs/config/pretrain/vision_transformer).
- **[January 2021]**: VISSL v0.1.5 released.

## Introduction
VISSL is a computer **VI**sion library for state-of-the-art **S**elf-**S**upervised **L**earning research with [PyTorch](https://pytorch.org). VISSL aims to accelerate research cycle in self-supervised learning: from designing a new self-supervised task to evaluating the learned representations. Key features include:

- **Reproducible implementation of SOTA in Self-Supervision**: All existing SOTA in Self-Supervision are implemented - [SwAV](https://arxiv.org/abs/2006.09882), [SimCLR](https://arxiv.org/abs/2002.05709), [MoCo(v2)](https://arxiv.org/abs/1911.05722), [PIRL](https://arxiv.org/abs/1912.01991), [NPID](https://arxiv.org/pdf/1805.01978.pdf), [NPID++](https://arxiv.org/abs/1912.01991), [DeepClusterV2](https://arxiv.org/abs/2006.09882), [ClusterFit](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yan_ClusterFit_Improving_Generalization_of_Visual_Representations_CVPR_2020_paper.pdf), [RotNet](https://arxiv.org/abs/1803.07728), [Jigsaw](https://arxiv.org/abs/1603.09246). Also supports supervised trainings.

- **Benchmark suite**: Variety of benchmarks tasks including [linear image classification (places205, imagenet1k, voc07, food, CLEVR, dsprites, UCF101, stanford cars and many more)](https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/linear_image_classification), [full finetuning](https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/fulltune), [semi-supervised benchmark](https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/semi_supervised), [nearest neighbor benchmark](https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/nearest_neighbor), [object detection (Pascal VOC and COCO)](https://github.com/facebookresearch/vissl/tree/master/configs/config/benchmark/object_detection).

- **Ease of Usability**: easy to use using yaml configuration system based on [Hydra](https://github.com/facebookresearch/hydra).

- **Modular**: Easy to design new tasks and reuse the existing components from other tasks (objective functions, model trunk and heads, data transforms, etc.). The modular components are simple *drop-in replacements* in yaml config files.

- **Scalability**: Easy to train model on 1-gpu, multi-gpu and multi-node. Several components for large scale trainings provided as simple config file plugs: [Activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html), [ZeRO](https://arxiv.org/abs/1910.02054), [FP16](https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use), [LARC](https://arxiv.org/abs/1708.03888), Stateful data sampler, data class to handle invalid images, large model backbones like [RegNets](https://arxiv.org/abs/2003.13678), etc.

- **Model Zoo**: Over *60 pre-trained self-supervised model* weights.

## Installation

See [`INSTALL.md`](https://github.com/facebookresearch/vissl/blob/master/INSTALL.md).

## Getting Started

Install VISSL by following the [installation instructions](https://github.com/facebookresearch/vissl/blob/master/INSTALL.md).
After installation, please see [Getting Started with VISSL](https://github.com/facebookresearch/vissl/blob/master/GETTING_STARTED.md) and the [Colab Notebook](https://colab.research.google.com/drive/1iigQmKL_DUuBLT6BqjrGXlW9ZIqKIFmt?usp=sharing) to learn about basic usage.

## Documentation

Learn more about VISSL at our [documentation](https://vissl.readthedocs.io). And see the [projects/](projects/) for some projects built on top of VISSL.

## Tutorials

Get started with VISSL by trying one of the **Colab tutorial notebooks**.

- [Train SimCLR on 1-gpu](https://colab.research.google.com/drive/1Rt3Plt3ph84i1A-eolLFafybwjrBFxYe?usp=sharing)
- [Extracting Features from a pretrained model](https://colab.research.google.com/drive/1EXZyW65lXBryWE2ZmC0bar2996KEiBZb?usp=sharing)
- [Benchmark task: Full finetuning on ImageNet-1K](https://colab.research.google.com/drive/1m1LUa-3vIR-rxwcm0QCrefc5S6PAY874?usp=sharing)
- [Benchmark task: Linear image classification on ImageNet-1K](https://colab.research.google.com/drive/1CCuZ50BN99JcOB6VEPytVi_i2tSMd7A3?usp=sharing)
- [Large scale training (fp16, LARC, ZeRO)](https://colab.research.google.com/drive/1fvZdRNUyHxMOaxuEO34x7XeGndzTUfIW?usp=sharing)
- [Using a pre-trained model in inference mode](https://colab.research.google.com/drive/1DkqGi9L47eWjA5j77EVEmbrktWBGoynN?usp=sharing)


## Model Zoo and Baselines
We provide a large set of baseline results and trained models available for download in the [VISSL Model Zoo](https://github.com/facebookresearch/vissl/blob/master/MODEL_ZOO.md).

## Contributors

VISSL is written and maintained by the Facebook AI Research.

## Development

We welcome new contributions to VISSL and we will be actively maintaining this library! Please refer to [`CONTRIBUTING.md`](./.github/CONTRIBUTING.md) for full instructions on how to run the code, tests and linter, and submit your pull requests.

## License

VISSL is released under [MIT license](LICENSE).

## Citing VISSL

If you find VISSL useful in your research or wish to refer to the baseline results published in the [Model Zoo](https://github.com/facebookresearch/vissl/blob/master/MODEL_ZOO.md), please use the following BibTeX entry.

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
