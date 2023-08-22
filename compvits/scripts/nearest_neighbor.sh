#!/bin/bash

M=$1
model=$2
K=$3
echo nearest_neighbor: M$M $model K$K

dir="logs/nearest_neighbor/M${M}/${model}/K$K"
feats_all="logs/nearest_neighbor/train_features/${model}"
feats_K="logs/extract_features/M${M}/${model}/K$K"

if [[ $model == "deitb" ]]; then
    trunk_cfg=deitb
else
    trunk_cfg=vitb
fi

mkdir --parents $dir
mv ${feats_all}/rank0_chunk0_train*.npy ${dir}
mv ${feats_K}/rank0_chunk0_test*.npy ${dir}

python tools/nearest_neighbor_test.py \
    config=compvits/base \
    +config/compvits/model/trunk=$trunk_cfg\
    +config/compvits/task=nearest_neighbor \
    +config/compvits/data/test=in1k \
    config.CHECKPOINT.DIR=$dir \
    config.NEAREST_NEIGHBOR.FEATURES.PATH=$dir \
    
mv ${dir}/rank0_chunk0_train*.npy ${feats_all}
mv ${dir}/rank0_chunk0_test*.npy ${feats_K}
