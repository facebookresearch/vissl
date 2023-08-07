#!/bin/bash

cm=$1
model=$2
K=$3

echo nearest_neighbor: $cm $model $K

dir="logs/nearest_neighbor/${cm}/${model}/K$K"
#feats="logs/nearest_neighbor/features/${model}"
feats="logs/extract_features/${cm}/${model}/K$K"

#echo extract_features: $cm $model $K
#dir=$feats
#python tools/run_distributed_engines.py \
#    config=compvits/vits_trunk \
#    +config/compvits/data/train=in1k_tiny \
#    config.TEST_MODEL=False \
#    config.CHECKPOINT.DIR=$dir \
#    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/ibot/vits_teacher.pth \
#    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=state_dict \
#    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk. \
#    config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False \
#    config.EXTRACT_FEATURES.OUTPUT_DIR=$dir \
#    engine_name=extract_features

mkdir --parents $dir
cp ${feats}/*.npy ${dir}
cp logs/nearest_neighbor/features/${model}/*.npy ${dr}

python tools/nearest_neighbor_test.py \
    config=compvits/vits_trunk \
    +config/compvits/data/test=in1k_tiny \
    +config/compvits/data/test/transforms=$cm \
    +config/compvits/benchmark=knn \
    config.TEST_ONLY=True \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/${model}/vits_teacher.pth \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=state_dict \
    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk. \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.NAME=afterK \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.PARAMS.K=$K \
    config.NEAREST_NEIGHBOR.FEATURES.PATH=$dir