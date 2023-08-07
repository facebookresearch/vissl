#!/bin/bash


cm=$1
model=$2
K=$3

echo extract_features: $cm $model $K
dir="logs/extract_features/${cm}/${model}/K$K"
python tools/run_distributed_engines.py \
    config=compvits/vits_trunk \
    +config/compvits/data/test=in1k_tiny \
    +config/compvits/data/test/transforms=$cm \
    config.TEST_ONLY=True \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/ibot/vits_teacher.pth \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=state_dict \
    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk. \
    config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.NAME=afterK \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.PARAMS.K=$K \
    config.EXTRACT_FEATURES.OUTPUT_DIR=${dir} \
    engine_name=extract_features