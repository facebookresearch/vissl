## Input Parameter Values for various datasets

The original ResNet-50 models from ICCV'19 in Caffe2.

- **RN50 Supervised IN1K**: /mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/prigoyal/20180618/rn50_baseline_p100_configs/baselines/IN1K_resnet50_8gpu_baseline.yaml.09_01_45.4vorasC6/imagenet1k/resnet_imagenet/depth_50/bestModel.pkl

- **RN50 Supervised Places-205**: /mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/prigoyal/20190222/rn50_places205_supervised_configs/baselines/places205_resnet50_8gpu_baseline.yaml.13_06_31.t6GazrHw/places205/resnet_imagenet/depth_50/bestModel.pkl

- **RN50 Jigsaw IN1K**: /mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/prigoyal/20181015/jigsaw_rn50_in1k_5k_perms_configs/unsupervised/jigsaw/pretext/resnet/IN1k_8gpu_siamese_BN_bias_decay_shift_scale.yaml.08_29_01.MkJ6IRPy/imagenet1k/resnet_jigsaw_siamese_2fc/depth_50/c2_model_iter450450.pkl

- **RN50 Jigsaw IN22K**: /mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/prigoyal/20181107/jigsaw_alexnet_pretext_in22k_90ep_5kperm_resume_configs/unsupervised/jigsaw/pretext/resnet/IN22k_8gpu_siamese_BN_bias_decay_shift_scale.yaml.06_29_42.F1h40tWH/imagenet22k/resnet_jigsaw_siamese_2fc/depth_50/c2_model_iter4989985.pkl

- **RN50 Jigsaw YFCC100M**: /mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/prigoyal/20181030/rn50_jigsaw_10kperm_yfcc100m_10ep_configs/unsupervised/jigsaw/pretext/resnet/in1k_schedule/yfcc100m_siamese_BN_bias_decay_shift_scale.yaml.06_30_10.90Ykc6Kn/yfcc100m/resnet_jigsaw_siamese_2fc/depth_50/c2_model_iter3903900.pkl

- **RN50 Colorization IN1K**: /mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/prigoyal/20180821/colorPre_rn50_in1k_24e-5_configs/unsupervised/colorization/pretext/resnet/resnet_IN1k_8gpu_colorize_BN_moment0.9_noBiasDecay_24e-5LR_noMagicInit.yaml.19_49_06.1yoRU8eI/imagenet1k/resnet_colorize/depth_50/c2_model_iter56056.pkl

- **RN50 Colorization IN22K**: /mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/prigoyal/20181017/color_rn50_in22k_84ep_configs/unsupervised/colorization/pretext/resnet/IN22k_8gpu_colorize_BN_moment0.9_noBiasDecay_24e-5LR_noMagicInit.yaml.06_15_13.w9OA1vLv/imagenet22k/resnet_colorize/depth_50/c2_model_iter1860000.pkl

- **RN50 Colorization low shot IN22K**: /mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/prigoyal/20181127/color_rn50_in22k_112ep_configs/unsupervised/colorization/pretext/resnet/IN22k_8gpu_colorize_BN_moment0.9_noBiasDecay_24e-5LR_noMagicInit.yaml.12_23_47.p8SHXwS9/imagenet22k/resnet_colorize/depth_50/c2_model_iter2480000.pkl

- **RN50 Colorization YFCC100M**: /mnt/vol/gfsai-bistro-east/ai-group/bistro/gpu/prigoyal/20181127/color_rn50_yfcc100m_30ep_configs/unsupervised/colorization/pretext/resnet/yfcc100m_IN1k_8gpu_colorize_BN_moment0.9_noBiasDecay_24e-5LR_noMagicInit.yaml.12_34_56.PlUKohdr/yfcc100m/resnet_colorize/depth_50/c2_model_iter4686682.pkl


The original AlexNet models from ICCV'19 in Caffe2 (Supervised, Jigsaw, Colorization) and RotNet/DeepCluster official models:

- **AlexNet Supervised IN1K**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/caffenet_bvlc_in1k_supervised.npy

- **AlexNet Supervised Places-205**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/caffenet_bvlc_places205_supervised.pkl

- **AlexNet Jigsaw IN1K**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/alexnet_jigsaw_in1k_pretext.pkl

- **AlexNet Jigsaw IN22K**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/alexnet_jigsaw_in22k_pretext.pkl

- **AlexNet Jigsaw YFCC100M**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/alexnet_jigsaw_yfcc100m_pretext.pkl

- **AlexNet Colorization IN1K**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/alexnet_colorization_in1k_pretext.pkl

- **AlexNet Colorization IN22K**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/alexnet_colorization_in22k_pretext.pkl

- **AlexNet Colorization YFCC100M**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/alexnet_colorization_yfcc100m_pretext.pkl

- **AlexNet DeepCluster IN1K**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/deepcluster_alexnet_checkpoint.pth.tar

- **AlexNet RotNet IN1K**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/iccv19_models/rotnet_alexnet_model_net_epoch50


The original NPID models from repo https://github.com/zhirongw/lemniscate.pytorch#pretrained-model:

- **ResNet-50 NPID IN1K model**: /mnt/vol/gfsai-bistro2-east/ai-group/users/prigoyal/sslime/converted_npid_oss_rn50_weights.npy


Torchvision ResNet-50 and ResNet-101 weights:

- **ResNet-50 Supervised IN1K**: /mnt/vol/gfsai-east/ai-group/users/prigoyal/sslime/converted_resnet50-19c8e357.pth

- **ResNet-101 Supervised IN1K**: /mnt/vol/gfsai-east/ai-group/users/prigoyal/sslime/converted_resnet101-5d3b4d8f.pth
