# SwAV in VISSL
**Unsupervised Learning of Visual Features by Contrasting Cluster Assignments**

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin

[[`SwAV`](https://github.com/facebookresearch/swav)] [[`arXiv`](https://arxiv.org/abs/2006.09882)] [[`BibTeX`](#CitingSwAV)]

<div align="center">
  <img width="100%" alt="SwAV Illustration" src="https://dl.fbaipublicfiles.com/deepcluster/animated.gif">
</div>

In this repository, we implement SwAV in VISSL.
To train a model, use the configs specified [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/pretrain/swav).

# Model Zoo

To use a pre-trained SwAV ResNet-50 model, simply do:
```python
import torch
model = torch.hub.load('facebookresearch/swav', 'resnet50')
```

We provide several baseline SwAV pre-trained models with ResNet-50 architecture in torchvision format.
We also provide models pre-trained with DeepCluster-v2 and SeLa-v2 obtained by applying improvements from the self-supervised community to [DeepCluster](https://arxiv.org/abs/1807.05520) and [SeLa](https://arxiv.org/abs/1911.05371) (see details in the [appendix of our paper](https://arxiv.org/abs/2006.09882)).

| method | epochs | batch-size | multi-crop | ImageNet top-1 acc. | url |
|-------------------|-------------------|---------------------|--------------------|--------------------|--------------------|
| SwAV | 800 | 4096 | 2x224 + 6x96 | 75.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar) |
| SwAV | 400 | 4096 | 2x224 + 6x96 | 74.6 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar) |
| SwAV | 200 | 4096 | 2x224 + 6x96 | 73.9 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_pretrain.pth.tar) |
| SwAV | 100 | 4096 | 2x224 + 6x96 | 72.1 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar) |
| SwAV | 200 | 256 | 2x224 + 6x96 | 72.7 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_bs256_pretrain.pth.tar) |
| SwAV | 400 | 256 | 2x224 + 6x96 | 74.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_bs256_pretrain.pth.tar) |
| SwAV | 400 | 4096 | 2x224 | 70.1 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_2x224_pretrain.pth.tar) |
| DeepCluster-v2 | 800 | 4096 | 2x224 + 6x96 | 75.2 | [model](https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar) |
| DeepCluster-v2 | 400 | 4096 | 2x160 + 4x96 | 74.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_400ep_pretrain.pth.tar) |
| DeepCluster-v2 | 400 | 4096 | 2x224 | 70.2 | [model](https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_400ep_2x224_pretrain.pth.tar) |
| SeLa-v2 | 400 | 4096 | 2x160 + 4x96 | 71.8 | [model](https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar) |
| SeLa-v2 | 400 | 4096 | 2x224 | 67.2 | [model](https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_2x224_pretrain.pth.tar) |

## Larger architectures
We provide SwAV models with ResNet-50 networks where we multiply the width by a factor ×2, ×4, and ×5.
| network | parameters | epochs | ImageNet top-1 acc. | url |
|-------------------|---------------------|--------------------|--------------------|--------------------|
| RN50-w2 | 94M | 400 | 77.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w2_400ep_pretrain.pth.tar) |
| RN50-w4 | 375M | 400 | 77.9 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w4_400ep_pretrain.pth.tar) |
| RN50-w5 | 586M | 400 | 78.5 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w5_400ep_pretrain.pth.tar) |

## <a name="CitingSwAV"></a>Citing SwAV

If you use SwAV, please use the following BibTeX entry.

```
@article{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
