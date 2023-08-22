#!/bin/bash

M=$1
model=$2
K=$3
echo test_linear: M$M $model K$K

dir="logs/test_linear/M${M}/${model}/K$K"

if [[ $model == "deitb" ]]; then
    head_cfg=mlp_768_1000
    trunk_cfg=deitb
else
    head_cfg=mlp_emlp_768_1000
    trunk_cfg=vitb
fi

python tools/run_distributed_engines.py \
    config=compvits/base \
    +config/compvits/model/trunk=$trunk_cfg \
    +config/compvits/model/head=$head_cfg \
    +config/compvits/data/test=in1k \
    +config/compvits/task=test_linear \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/${model}.pth \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.COMP.NAME=afterK \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.COMP.PARAMS.K=$K \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.SPLIT.NAME=precomputed_masks \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.SPLIT.PARAMS.M=$M \
    