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

### NPID++
- /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_cosineLR_ep800_4node.torch
- /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_cosineLR_ep800_8gpu.torch
- /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_stepLR_ep800_8gpu.torch
- /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg4k_stepLR_ep200_4node.torch
- /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg4k_stepLR_ep200_8gpu.torch
- /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w2_npid_neg4k_stepLR_ep200_4node.torch
- /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w2_npid_neg4k_stepLR_ep800_4node.torch
- /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w3_npid_neg32k_stepLR_ep800_4node.torch
- /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w3_npid_neg4k_stepLR_ep200_4node.torch


### PIRL
<!-- PIRL -->
- rn50 - in1k - jigsaw - ep200 [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_pirl_in1k_jigsaw_ep200.torch
- rn50 - in1k - jigsaw - ep800 [old] - /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_pirl_in1k_jigsaw_ep800.torch
- rn50 - in1k - jigsaw - ep200 -  TODO
- rn50 - in1k - jigsaw - ep800 -  TODO

### SimCLR
<!-- simclr -->
- rn50 - in1k - 100ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_100ep_simclr_8node_resnet_16_07_20.8edb093e/model_final_checkpoint_phase99.torch
- rn50 - in1k - 200ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_200ep_simclr_8node_resnet_16_07_20.a816c0ef/model_final_checkpoint_phase199.torch
- rn50 - in1k - 400ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_400ep_simclr_8node_resnet_16_07_20.36b338ef/model_final_checkpoint_phase399.torch
- rn50 - in1k - 800ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch
- rn50 - in1k - 1000ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch
- rn50w2 - in1k - 100ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50w2_100ep_simclr_8node_resnet_16_07_20.05b37ec3/model_final_checkpoint_phase99.torch
- rn50w2 - in1k - 1000ep
- rn50w4 - in1k - 1000ep
- rn101 - in1k - 100ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn101_100ep_simclr_8node_resnet_16_07_20.1ff6cb4b/model_final_checkpoint_phase99.torch
- rn101 - in1k - 1000ep

### SwAV
<!-- swav -->
- rn50 - 2x224 + 6x96 - 100ep - 8node:
- rn50 - 2x224 + 6x96 - 200ep - 8node:
- rn50 - 2x224 + 6x96 - 400ep - 8node:
- rn50 - 2x224 + 6x96 - 800ep - 8node:
- rn50 - 2x224 + 6x96 - 200ep - 1node - use queue:
- rn50 - 2x224 + 6x96 - 400ep - 1node - use queue:
- rn50 - 2x224 - 400ep - 8node:
- rn50w2 - 2x224 + 6x96 - 400ep - 16node - bs32:
- rn50w4 - 2x224 + 6x96 - 400ep - 8node - bs40:
- rn50w5 - 2x224 + 6x96 - 400ep - 16node - bs12:
- rn50w2 - 2x224 + 6x96 - 400ep - 16node - bs32: /checkpoint/imisra/dcluster2/checkpoints_only/r50w2_swav_2x224_4x96v3_mlp8k_lr4pt8_minlr1pt-3_bg32_ep400/model_final_checkpoint_phase399.torch
- rn50w4 - 2x224 + 6x96 - 400ep - 8node - bs40: /checkpoint/imisra/dcluster2/mathilde_runs/pretext_rn50w4_b40wq_2x224_4x96v2_indep_1head_8nodes_oto_imagenet_ampO1_minlr1pt-3_ep400/checkpoints/model_final_checkpoint_phase399.torch
- rn50w5 - 2x224 + 6x96 - 400ep - 16node - bs12: /checkpoint/imisra/dcluster2/mathilde_runs/pretext_rn50w5_b12wq_2x224_4x96v3_indep_noBN_1head_16nodes_oto_imagenet_ampO1_ep400/checkpoints/model_final_checkpoint_phase399.torch

### DeepClusterV2
<!-- deepcluster v2 -->
- rn50 - 2x160 + 4x96 - 400ep - 8node -
- rn50 - 2x224 - 400ep - 8node -

### SeLA-V2
<!-- sela-v2 -->
- rn50 - 2x224 - 400ep - 8node - https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_2x224_pretrain.pth.tar
- rn50 - 2x224 + 4x96 - 400ep - 8node - https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar)

### MoCo
<!-- moco -->
