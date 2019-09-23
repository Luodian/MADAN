#!/usr/bin/env bash
# Sequentially load two generators(GTA, Synthia) and finish
how_many=100000

cd /root/MADAN/cyclegan
model=$1
epoch=$2

python3 test.py --name ${model} --resize_or_crop=None \
    --which_model_netD n_layers --n_layers_D 3 \
    --model $3 \
    --no_flip --batchSize 32 --nThreads 16 \
    --dataset_mode $4 --dataroot /nfs/project/libo_i/cycada/data \
    --which_direction AtoB \
    --phase train --out_all \
    --how_many ${how_many} --which_epoch ${epoch} --gpu 0


python3 test.py --name ${model} --resize_or_crop=None \
    --which_model_netD n_layers --n_layers_D 3 \
    --model $3 \
    --no_flip --batchSize 32 --nThreads 16 \
    --dataset_mode $5 --dataroot /nfs/project/libo_i/cycada/data \
    --which_direction AtoB \
    --phase train --out_all \
    --how_many ${how_many} --which_epoch ${epoch} --gpu 0

# cyclegan/test_templates_cycle.sh cycada_gta_synthia2cityscapes_noIdentity_D12D21D3_SEM_final_scale 15 test synthia_cityscapes gta5_cityscapes