#!/bin/bash


model=$1
echo train_linear: $model
dir=train_debug

python tools/run_distributed_engines.py \
    config=compvits/vits_trunk \
    +config/compvits/data/train=in1k_tiny \
    +config/compvits/data/test=in1k_tiny \
    +config/compvits/benchmark=train_linear \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/ibot/vits_teacher.pth \
    config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=state_dict \
    config.MODEL.WEIGHTS_INIT.APPEND_PREFIX=trunk. \
    