#!/bin/bash

M=$1
model=$2
K=$3

echo extract_features: M$M $model K$K
dir="logs/extract_features/M${M}/${model}/K$K"

if [[ $model == "deitb" ]]; then
    trunk_cfg=deitb
else
    trunk_cfg=vitb
fi

python tools/run_distributed_engines.py \
    config=compvits/base \
    +config/compvits/model/trunk=$trunk_cfg \
    +config/compvits/data/test=in1k \
    engine_name=extract_features \
    config.TEST_ONLY=True \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/${model}.pth \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.COMP.NAME=afterK \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.COMP.PARAMS.K=$K \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.SPLIT.NAME=precomputed_masks \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.SPLIT.PARAMS.M=$M \
    