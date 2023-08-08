#!/bin/bash

script=$1

models=(ibot)
cms=configs/config/compvits/data/test/transforms/*.yaml

for cm in $cms; do
    cm=$(basename $cm .yaml)
    for model in ${models[@]}; do
        for ((K=0; K<=8; K++)); do
            source $script $cm $model $K
        done
    done
done