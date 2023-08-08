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
    config.TEST_ONLY=True \
    config.CHECKPOINT.DIR=$dir \
    config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/jan.olszewski/git/vissl/checkpoints/ibot/vits_teacher_linear.pth \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.NAME=afterK \
    config.MODEL.TRUNK.VISION_TRANSFORMERS.COMPVITS.PARAMS.K=$K \