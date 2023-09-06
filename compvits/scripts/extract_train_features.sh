#!/bin/bash

model=$1

echo extract_train_features: $model

dir="logs/nearest_neighbor/test_features/${model}"

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
    config.TEST_MODEL=False \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=checkpoints/trunk_only/${model}.pth \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=model \
