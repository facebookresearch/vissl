# VISSL Model Zoo and Benchmarks

## Introduction

VISSL provides reference implementation of a large number of self-supervision approaches and also a suite of benchmark tasks to quickly evaluate the representation quality of models trained with these self-supervised tasks using standard evaluation setup. In this document, we list the collection of self-supervises models and benchmarks of these models on a subset of benchmark suite. All the models can be downloaded from the provided link.

### Torchvision models

VISSL is 100% compatible with TorchVision ResNet models. You can benchmark these models using VISSL's benchmark suite. See the docs for how to run various benchmarks.

|------------|-------------| Linear Classification                 | Object Detection       | Full finetuning |
! Model      | Description | VOC07 | Places205 | iNat18 | ImageNet | VOC07+12 (Faster-RCNN) | ImageNet        |


### Supervised ImageNet pre-trained models

rn50 - in1k - caffe2
rn50 - in1k - vissl
rn50 - in1k - torchvision
rn50 - places205 - caffe2
alexnet - bvlc - caffe2
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_in1k_caffe2.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_in1k_vissl.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_places205_caffe2.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_torchvision_19c8e357.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_caffenet_bvlc_in1k_supervised.torch

### Semi-weakly supervised models
rn50 - semi sup - zeki
rn50 - semi weakly sup - zeki
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_semi_sup_08389792.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_semi_weakly_sup_16a12f1b.torch

### RotNet

RN50 - in1k
RN50 - in22k
alexnet - oss
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_rotnet_in1k_ep90.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_rotnet_in22k_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/rotnet_alexnet_model_net_epoch50.torch


### Jigsaw

RN50 - perm100 - in1k
RN50 - perm10k - in1k
RN50 - perm2k - in1k
RN50 - perm2k - in22k - 4node
RN50 - perm2k - in22k - 8gpu
RN50 - in1k - goyal19
RN50 - in22k - goyal19
RN50 - yfcc100m - goyal19
alexnet - in1k - goyal19
alexnet - in22k - goyal19
alexnet - yfcc100m - goyal19
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm100_in1k_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm10k_in1k_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm2k_in1k_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm2k_in22k_4node_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm2k_in22k_8gpu_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_jigsaw_in1k_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_jigsaw_in22k_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_jigsaw_yfcc100m_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_jigsaw_in1k_pretext.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_jigsaw_in22k_pretext.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_jigsaw_yfcc100m_pretext.torch

### Colorization

rn50 - in1k - iccv19
rn50 - in22k - iccv19
rn50 - yfcc100m - iccv19
alexnet - in1k - iccv19
alexnet - in22k - iccv19
alexnet - yfcc100m - iccv19

/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_colorization_in1k_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_colorization_in22k_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_colorization_yfcc100m_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_colorization_in1k_pretext.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_colorization_in22k_pretext.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_colorization_yfcc100m_pretext.torch

### DeepCluster

alexnet - repo
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/deepcluster_alexnet_checkpoint.torch

### NPID

RN50 - in1k - 8gpu - lemniscate
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_lemniscate_neg4k_stepLR_8gpu.torch


### NPID++

/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_cosineLR_ep800_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_cosineLR_ep800_8gpu.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_stepLR_ep800_8gpu.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg4k_stepLR_ep200_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg4k_stepLR_ep200_8gpu.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w2_npid_neg4k_stepLR_ep200_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w2_npid_neg4k_stepLR_ep800_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w3_npid_neg32k_stepLR_ep800_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w3_npid_neg4k_stepLR_ep200_4node.torch


### ClusterFit

rn50 - 16k clusters - in1k - ep105
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_rotnet_16kclusters_in1k_ep105.torch

### PIRL

rn50 - in1k - jigsaw - ep200
rn50 - in1k - jigsaw - ep800
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_pirl_in1k_jigsaw_ep200.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_pirl_in1k_jigsaw_ep800.torch

### SimCLR

rn50 - in1k - 100ep
rn50 - in1k - 200ep
rn50 - in1k - 400ep
rn50 - in1k - 800ep
rn50 - in1k - 1000ep
rn50w2 - in1k - 100ep
rn50w2 - in1k - 400ep
rn50w2 - in1k - 1000ep
rn50w3 - in1k - 100ep
rn50w3 - in1k - 1000ep
rn50w4 - in1k - 100ep
rn50w4 - in1k - 1000ep
rn101 - in1k - 100ep
rn101 - in1k - 1000ep

### MoCo

### DeepCluster V2

### SwAV





<!-- Deepcluster -->
alexnet - repo
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/deepcluster_alexnet_checkpoint.torch


<!-- Jigsaw -->

RN50 - perm100 - in1k
RN50 - perm10k - in1k
RN50 - perm2k - in1k
RN50 - perm2k - in22k - 4node
RN50 - perm2k - in22k - 8gpu
RN50 - in1k - goyal19
RN50 - in22k - goyal19
RN50 - yfcc100m - goyal19
alexnet - in1k - goyal19
alexnet - in22k - goyal19
alexnet - yfcc100m - goyal19
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm100_in1k_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm10k_in1k_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm2k_in1k_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm2k_in22k_4node_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_jigsaw_rn50_perm2k_in22k_8gpu_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_jigsaw_in1k_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_jigsaw_in22k_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_jigsaw_yfcc100m_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_jigsaw_in1k_pretext.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_jigsaw_in22k_pretext.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_jigsaw_yfcc100m_pretext.torch

<!-- Colorization -->

rn50 - in1k - iccv19
rn50 - in22k - iccv19
rn50 - yfcc100m - iccv19
alexnet - in1k - iccv19
alexnet - in22k - iccv19
alexnet - yfcc100m - iccv19

/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_colorization_in1k_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_colorization_in22k_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_colorization_yfcc100m_goyal19.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_colorization_in1k_pretext.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_colorization_in22k_pretext.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_alexnet_colorization_yfcc100m_pretext.torch

<!-- NPID -->

RN50 - in1k - 8gpu - lemniscate
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_lemniscate_neg4k_stepLR_8gpu.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_cosineLR_ep800_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_cosineLR_ep800_8gpu.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg32k_stepLR_ep800_8gpu.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg4k_stepLR_ep200_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_npid_neg4k_stepLR_ep200_8gpu.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w2_npid_neg4k_stepLR_ep200_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w2_npid_neg4k_stepLR_ep800_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w3_npid_neg32k_stepLR_ep800_4node.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_w3_npid_neg4k_stepLR_ep200_4node.torch

<!-- PIRL -->

rn50 - in1k - jigsaw - ep200
rn50 - in1k - jigsaw - ep800
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_pirl_in1k_jigsaw_ep200.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_pirl_in1k_jigsaw_ep800.torch

<!-- ClusterFit -->
rn50 - 16k clusters - in1k - ep105
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_rotnet_16kclusters_in1k_ep105.torch

<!-- RotNet -->
RN50 - in1k
RN50 - in22k
alexnet - oss
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_rotnet_in1k_ep90.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_rotnet_in22k_ep105.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/rotnet_alexnet_model_net_epoch50.torch

<!-- Semi-weakly -->
rn50 - semi sup - zeki
rn50 - semi weakly sup - zeki
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_semi_sup_08389792.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_semi_weakly_sup_16a12f1b.torch

<!-- Supervised -->

rn50 - in1k - caffe2
rn50 - in1k - vissl
rn50 - in1k - torchvision
rn50 - places205 - caffe2
alexnet - bvlc - caffe2
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_in1k_caffe2.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_in1k_vissl.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_places205_caffe2.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_supervised_torchvision_19c8e357.torch
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/sslime/converted_caffenet_bvlc_in1k_supervised.torch

<!-- URU -->
rn50 - ig1billion
/mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/vissl/converted_vissl_rn50_uru_ig1billion_in_labelspace_nonclustered_1477labels.torch

<!-- simclr -->
rn50 - in1k - 100ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_100ep_simclr_8node_resnet_16_07_20.8edb093e/model_final_checkpoint_phase99.torch
rn50 - in1k - 200ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50_200ep_simclr_8node_resnet_16_07_20.a816c0ef/model_final_checkpoint_phase199.torch
rn50 - in1k - 400ep
rn50 - in1k - 800ep
rn50 - in1k - 1000ep
rn50w2 - in1k - 100ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn50w2_100ep_simclr_8node_resnet_16_07_20.05b37ec3/model_final_checkpoint_phase99.torch
rn50w2 - in1k - 1000ep
rn50w3 - in1k - 100ep
rn50w3 - in1k - 1000ep
rn50w4 - in1k - 100ep /mnt/vol/gfsai-bistro2-east/ai-group/bistro/gpu/prigoyal/ssl_framework/simclr_rn101_100ep_simclr_8node_resnet_16_07_20.1ff6cb4b/model_final_checkpoint_phase99.torch
rn50w4 - in1k - 1000ep
rn101 - in1k - 100ep
rn101 - in1k - 1000ep

<!-- swav -->
to check with mathilde

<!-- moco -->

<!-- deepcluster v2 -->
