#!/bin/bash

./dev/launch_slurm.sh \
    config=pretrain/byol/byol_1node_resnet \
    config.SLURM.NAME=byol_test \
    config.SLURM.COMMENT="BYOL FOR VISSL" \
    config.SLURM.PARTITION=learnfair \
