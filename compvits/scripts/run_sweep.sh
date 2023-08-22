#!/bin/bash

#models=(ibot)
#cms=configs/config/compvits/data/test/transforms/*.yaml

#for cm in $cms; do
#    cm=$(basename $cm .yaml)
#    for model in ${models[@]}; do
#        for ((K=0; K<=8; K++)); do
#            source $script $cm $model $K
#        done
#    done
#done

source compvits/scripts/extract_train_features.sh

scripts=(compvits/scripts/test_linear.sh compvits/scripts/extract_features.sh compvits/scripts/nearest_neighbor.sh)
models=(deitb)
Ms=(2 3 4 6 8 9 12 16)
for script in ${scripts[@]}; do
    for M in ${Ms[@]}; do
        for model in ${models[@]}; do
            for ((K=0; K<=12; K++)); do
                source $script $M $model $K
            done
        done
    done
done