# PIRL
**Self-Supervised Learning of Pretext-Invariant Representations**

Ishan Misra, Laurens van der Maaten

[[`arXiv`](https://arxiv.org/abs/1912.01991)] [[`BibTeX`](#Citation)]

<div align="left">
  <img width="50%" alt="PIRL_teaser_figure" src="http://imisra.github.io/data/teaser-imgs/pirl_teaser.jpg">
</div>


# Training
All the model configs used for training models are found under the `configs/config/pretrain/pirl` directory [here](https://github.com/facebookresearch/vissl/tree/main/configs/config/pretrain/pirl).

For example, to train a ResNet-50 model used in the PIRL paper, you can run

```
python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50
```

## Improvements to PIRL training
We can train the PIRL model with improvements from SimCLR (Chen et al., 2020), namely - the MLP head for projection of features and the Gaussian blur data augmentations.

```
python tools/run_distributed_engines.py config=pretrain/pirl/pirl_jigsaw_4node_resnet50 \
    +config/pretrain/pirl/models=resnet50_mlphead
    +config/pretrain/pirl/transforms=photo_gblur
```

# Model Zoo

We provide the following pretrained models and report their single crop top-1 accuracy on the ImageNet validation set.

| Model | Epochs | Head | Top-1 |  Checkpoint |
| ----- | ------ | -----| ----- | ----------- |
| R50 |  200 | Linear | 62.9 |  [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/r50_200ep/model_final_checkpoint_phase199.torch) |
| R50 | 200 | MLP | 65.8 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/r50_200ep_mlp_gblur/model_final_checkpoint_phase199.torch) |
| R50 | 800 | Linear | 64.29 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch) |
| R50 | 800 | MLP | 69.9 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/r50_800ep_mlphead_gblur/model_final_checkpoint_phase799.torch) |
| R50w2 | 400 | Linear | 69.3 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/w2_400ep/model_final_checkpoint_phase399.torch) |
| R50w2 | 400 | MLP | 70.9 | [model](https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/r50w2_400ep_mlphead_gblur/model_final_checkpoint_phase399.torch) |


## <a name="Citation"></a>Citing PIRL

If you find PIRL useful, please consider citing the following paper
```
@inproceedings{misra2020pirl,
  title={Self-Supervised Learning of Pretext-Invariant Representations},
  author={Misra, Ishan and van der Maaten, Laurens},
  booktitle={CVPR},
  year={2020}
}
```
