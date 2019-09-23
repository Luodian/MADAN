#!/usr/bin/env bash

how_many=100000

cd /root/MADAN/cyclegan
name=$1
epoch=$2

python3 test.py --name ${name} --resize_or_crop=None \
    --which_model_netD n_layers --n_layers_D 3 \
    --model $3 --loadSize 600 \
    --no_flip --batchSize 32 --nThreads 16 \
    --dataset_mode $4 --dataroot /nfs/project/libo_i/cycada/data \
    --which_direction AtoB \
    --phase train --out_all \
    --how_many ${how_many} --which_epoch ${epoch} --gpu 0