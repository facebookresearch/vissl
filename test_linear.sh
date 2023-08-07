#!/bin/bash

#cm=cm_98_98_196_0
#model=ibot
#K=0

cm=$1
model=$2
K=$3

echo test_linear: $cm $model $K

dir="logs/test_linear/${cm}/${model}/K$K"
python tools/run_distributed_engines.py \
    config=compvits/vits_trunk \
    +config/compvits/data/test=in1k_tiny \
    +config/compvits/data/test/transforms=$cm \
    +config/compvits/benchmark=test_linear \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/ibot/vits_teacher_linear.pth \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.NAME=afterK \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.PARAMS.K=$K \

#K=0
#dir="logs/train_linear/K$K"
#python tools/run_distributed_engines.py \
#    config=compvits/compvit_s_trunk \
#    +config/compvits/data=in1k_tiny_train \
#    +config/compvits/benchmark=train_linear \
#    config.CHECKPOINT.DIR=$dir \
#    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/ibot/vit_s_teacher.pth \
#    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=state_dict \
#    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk. \
#    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.PARAMS.K=$K


#K=0
#knn_dir="logs/knn"
#dir="${knn_dir}/K$K"
#python tools/nearest_neighbor_test.py \
#    config=compvits/compvit_s_trunk \
#    +config/compvits/data=in1k_tiny_train \
#    +config/compvits/benchmark=knn \
#    config.CHECKPOINT.DIR=$dir \
#    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/ibot/vit_s_teacher.pth \
#    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=state_dict \
#    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk. \
#    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.PARAMS.K=$K \
#    config.NEAREST_NEIGHBOR.FEATURES=$knn_dir


#K=0
#knn_dir="logs/knn"
#dir="${knn_dir}/K$K"
#python tools/nearest_neighbor_test.py \
#    config=compvits/compvit_s_trunk \
#    +config/compvits/data=in1k_tiny_train \
#    +config/compvits/benchmark=knn \
#    config.CHECKPOINT.DIR=$dir \
#    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/ibot/vit_s_teacher.pth \
#    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=state_dict \
#    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk. \
#    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.PARAMS.K=$K \
#    config.NEAREST_NEIGHBOR.FEATURES=$knn_dir