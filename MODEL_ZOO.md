# VISSL Model Zoo and Benchmarks

## Introduction

VISSL provides reference implementation of a large number of self-supervision approaches and also a suite of benchmark tasks to quickly evaluate the representation quality of models trained with these self-supervised tasks using standard evaluation setup. In this document, we list the collection of self-supervises models and benchmarks of these models on a subset of benchmark suite. All the models can be downloaded from the provided link.

### Torchvision models

VISSL is 100% compatible with TorchVision ResNet models. You can benchmark these models using VISSL's benchmark suite. See the docs for how to run various benchmarks.

|------------|-------------| Linear Classification                 | Object Detection       | Full finetuning |
! Model      | Description | VOC07 | Places205 | iNat18 | ImageNet | VOC07+12 (Faster-RCNN) | ImageNet        |



### Sypervised
<!-- Supervised -->
- rn50 - in1k - caffe2 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_in1k_caffe2.torch
- rn50 - places205 - caffe2 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_places205_caffe2.torch
- alexnet - bvlc - caffe2 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_caffenet_bvlc_in1k_supervised.torch
- rn50 - in1k - vissl [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_in1k_vissl.torch
- rn50 - in1k - vissl - /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/sup_rn50_in1k_ep105_supervised_8gpu_resnet_17_07_20.733dbdee/model_final_checkpoint_phase208.torch
- rn50 - in1k - torchvision - https://download.pytorch.org/models/resnet50-19c8e357.pth
- rn101 - in1k - torchvision - https://download.pytorch.org/models/resnet101-5d3b4d8f.pth

### Semi-weakly supervised
<!-- Semi-weakly -->
- rn50 - semi sup - zeki - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_semi_sup_08389792.torch
- rn50 - semi weakly sup - zeki - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_semi_weakly_sup_16a12f1b.torch

### URU
<!-- URU -->
- rn50 - ig1billion - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_uru_ig1billion_in_labelspace_nonclustered_1477labels.torch

### Clusterfit
<!-- ClusterFit -->
- rn50 - 16k clusters - in1k - ep105 [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_rotnet_16kclusters_in1k_ep105.torch
- rn50 - 16k clusters - in1k - ep105 - /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/clusterfit_in1k_rotnet_16kclusters_clusterfit_resnet_8gpu_imagenet_20_07_20.42dfa942/model_final_checkpoint_phase104.torch

### RotNet
<!-- RotNet -->
- RN50 - in1k - ep90 [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_rotnet_in1k_ep90.torch
- RN50 - in1k - ep105 - /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch
- RN50 - in22k - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_rotnet_in22k_ep105.torch
- alexnet - oss - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/rotnet_alexnet_model_net_epoch50.torch

### Deepcluster
<!-- Deepcluster -->
- alexnet - repo - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/deepcluster_alexnet_checkpoint.torch

### Jigsaw
<!-- Jigsaw -->
- RN50 - perm100 - in1k [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm100_in1k_ep105.torch
- RN50 - perm2K - in1k [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm2k_in1k_ep105.torch
- RN50 - perm10K - in1k [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm10k_in1k_ep105.torch
- RN50 - perm100 - in1k - /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.db174a43/model_final_checkpoint_phase104.torch
- RN50 - perm10k - in1k - /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_20_07_20.3d706467/model_final_checkpoint_phase104.torch
- RN50 - perm2k - in1k - /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/jigsaw_rn50_in1k_ep105_perm2k_jigsaw_8gpu_resnet_17_07_20.cccee144/model_final_checkpoint_phase104.torch
- RN50 - perm2k - in22k - 4node - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm2k_in22k_4node_ep105.torch
- RN50 - perm2k - in22k - 8gpu - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm2k_in22k_8gpu_ep105.torch
- RN50 - in1k - goyal19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_jigsaw_in1k_goyal19.torch
- RN50 - in22k - goyal19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_jigsaw_in22k_goyal19.torch
- RN50 - yfcc100m - goyal19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_jigsaw_yfcc100m_goyal19.torch
- alexnet - in1k - goyal19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_jigsaw_in1k_pretext.torch
- alexnet - in22k - goyal19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_jigsaw_in22k_pretext.torch
- alexnet - yfcc100m - goyal19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_jigsaw_yfcc100m_pretext.torch

### Colorization
<!-- Colorization -->
- rn50 - in1k - iccv19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_colorization_in1k_goyal19.torch
- rn50 - in22k - iccv19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_colorization_in22k_goyal19.torch
- rn50 - yfcc100m - iccv19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_colorization_yfcc100m_goyal19.torch
- alexnet - in1k - iccv19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_colorization_in1k_pretext.torch
- alexnet - in22k - iccv19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_colorization_in22k_pretext.torch
- alexnet - yfcc100m - iccv19 - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_colorization_yfcc100m_pretext.torch


### NPID
<!-- NPID -->
- RN50 - in1k - 8gpu - lemniscate - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_lemniscate_neg4k_stepLR_8gpu.torch
- RN50 -in1k - 8gpu - 4kneg - 200ep - vissl: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/npid_1node_200ep_4kneg_npid_8gpu_resnet_23_07_20.9eb36512/model_final_checkpoint_phase199.torch

### NPID++
<!-- NPID++ -->
- RN50 - in1k - 4node - 32K neg - 800ep - cosine - vissl [old]: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_cosineLR_ep800_4node.torch
- RN50 - in1k - 4node - 32K neg - 800ep - cosine - vissl: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/npid_4node_800ep_32kneg_cosine_npid++_4nodes_resnet_23_07_20.75432662/model_final_checkpoint_phase799.torch
- RN50w2 - in1k - 4node - 32K neg - 800ep - cosine - vissl: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/npid_4node_800ep_32kneg_cosine_rn50w2_npid++_4nodes_resnet_27_07_20.b7f4016c/model_final_checkpoint_phase799.torch


### PIRL
<!-- PIRL -->
- rn50 - in1k - jigsaw - ep200 [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_pirl_in1k_jigsaw_ep200.torch
- rn50 - in1k - jigsaw - ep800 [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_pirl_in1k_jigsaw_ep800.torch
- rn50 - in1k - jigsaw - ep200 -  /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/pirl_jigsaw_4node_200ep_pirl_jigsaw_4node_resnet_22_07_20.ffd17b75/model_final_checkpoint_phase199.torch
- rn50 - in1k - jigsaw - ep800 -  /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch
<!-- PIRL trained by Ishan -->
- R50-200ep-62.85: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/imisra/ssl_framework/pirl/r50_200ep/model_final_checkpoint_phase199.torch
- R50-200ep-MLP-65.8: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/imisra/ssl_framework/pirl/r50_200ep_mlp_gblur/model_final_checkpoint_phase199.torch
- R50-800ep-Linear-63.8: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/imisra/ssl_framework/pirl/pretext_rn50_4nodes_pirl_imagenet_ep800/model_final_checkpoint_phase799.torch
- R50-800ep-MLP-69.9: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/imisra/ssl_framework/pirl/r50_800ep_mlphead_gblur/model_final_checkpoint_phase799.torch
- R50w2-400ep-Linear-69.3: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/imisra/ssl_framework/pirl/w2_400ep/model_final_checkpoint_phase399.torch
- R50w2-400ep-MLP-70.9: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/imisra/ssl_framework/pirl/r50w2_400ep_mlphead_gblur/model_final_checkpoint_phase399.torch

### SimCLR
<!-- simclr -->
- rn50 - in1k - 100ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_100ep_simclr_8node_resnet_16_07_20.8edb093e/model_final_checkpoint_phase99.torch
- rn50 - in1k - 200ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_200ep_simclr_8node_resnet_16_07_20.a816c0ef/model_final_checkpoint_phase199.torch
- rn50 - in1k - 400ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_400ep_simclr_8node_resnet_16_07_20.36b338ef/model_final_checkpoint_phase399.torch
- rn50 - in1k - 800ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch
- rn50 - in1k - 1000ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch
- rn50w2 - in1k - 100ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50w2_100ep_simclr_8node_resnet_16_07_20.05b37ec3/model_final_checkpoint_phase99.torch
- rn50w2 - in1k - 1000ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50w2_1000ep_simclr_8node_resnet_16_07_20.e1e3bbf0/model_final_checkpoint_phase999.torch
- rn50w4 - in1k - 1000ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50w4_1000ep_bs32_16node_simclr_8node_resnet_28_07_20.9e20b0ae/model_final_checkpoint_phase999.torch
- rn101 - in1k - 100ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn101_100ep_simclr_8node_resnet_16_07_20.1ff6cb4b/model_final_checkpoint_phase99.torch
- rn101 - in1k - 1000ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch

### SwAV
<!-- swav -->
- rn50 - 2x224 + 6x96 - 100ep - 8node: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/swav_in1k_rn50_100ep_swav_8node_resnet_27_07_20.7e6fc6bf/model_final_checkpoint_phase99.torch
- rn50 - 2x224 + 6x96 - 200ep - 8node: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/swav_in1k_rn50_200ep_swav_8node_resnet_27_07_20.bd595bb0/model_final_checkpoint_phase199.torch
- rn50 - 2x224 + 6x96 - 400ep - 8node: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/swav_in1k_rn50_400ep_swav_8node_resnet_27_07_20.a5990fc9/model_final_checkpoint_phase399.torch
- rn50 - 2x224 + 6x96 - 800ep - 8node: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/swav_in1k_rn50_800ep_swav_8node_resnet_27_07_20.a0a6b676/model_final_checkpoint_phase799.torch
- rn50 - 2x224 + 6x96 - 200ep - 1node - use queue: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/swav_4gpu_bs64_200ep_2x224_6x96_queue_swav_8node_resnet_28_07_20.a8f2c735/model_final_checkpoint_phase199.torch
- rn50 - 2x224 + 6x96 - 400ep - 1node - use queue: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/swav_4gpu_bs64_400ep_2x224_6x96_queue_swav_8node_resnet_28_07_20.5e967ca0/model_final_checkpoint_phase399.torch
- rn50 - 2x224 - 400ep - 8node: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/swav_8node_2x224_rn50_in1k_swav_8node_resnet_30_07_20.c8fd7169/model_final_checkpoint_phase399.torch
- rn50w2 - 2x224 + 6x96 - 400ep - 16node - bs32: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/swav_rn50w2_in1k_bs32_16node_400ep_swav_8node_resnet_30_07_20.93563e51/model_final_checkpoint_phase399.torch
- rn50w4 - 2x224 + 6x96 - 400ep - 8node - bs40: /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/swav_rn50w4_in1k_bs40_8node_400ep_swav_8node_resnet_30_07_20.1736135b/model_final_checkpoint_phase399.torch
- rn50w5 - 2x224 + 6x96 - 400ep - 16node - bs12: [TODO]
- rn50w2 - 2x224 + 6x96 - 400ep - 16node - bs32: /checkpoint/imisra/dcluster2/checkpoints_only/r50w2_swav_2x224_4x96v3_mlp8k_lr4pt8_minlr1pt-3_bg32_ep400/model_final_checkpoint_phase399.torch
- rn50w4 - 2x224 + 6x96 - 400ep - 8node - bs40: /checkpoint/imisra/dcluster2/mathilde_runs/pretext_rn50w4_b40wq_2x224_4x96v2_indep_1head_8nodes_oto_imagenet_ampO1_minlr1pt-3_ep400/checkpoints/model_final_checkpoint_phase399.torch
- rn50w5 - 2x224 + 6x96 - 400ep - 16node - bs12: /checkpoint/imisra/dcluster2/mathilde_runs/pretext_rn50w5_b12wq_2x224_4x96v3_indep_noBN_1head_16nodes_oto_imagenet_ampO1_ep400/checkpoints/model_final_checkpoint_phase399.torch
- swav official rn50-100ep-8node-8crop-in1k: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/swav_100ep_pretrain.pth.tar

### DeepClusterV2
<!-- deepcluster v2 -->
- rn50 - 2x160 + 4x96 - 400ep - 8node - [TODO]
- rn50 - 2x224 - 400ep - 8node - [TODO]

### SeLA-V2
<!-- sela-v2 -->
- rn50 - 2x224 - 400ep - 8node - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/selav2_400ep_2x224_pretrain.pth.tar
- rn50 - 2x224 + 4x96 - 400ep - 8node - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/selav2_400ep_pretrain.pth.tar

### MoCo
<!-- moco -->
