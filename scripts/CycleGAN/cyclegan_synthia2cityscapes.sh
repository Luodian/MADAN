#!/usr/bin/env bash
cd /root/MADAN/cyclegan

python3 train.py --name cycada_gta_synthia2cityscapes_noIdentity_D12D21D3_SEM_final_scale \
    --resize_or_crop scale_width_and_crop --loadSize 500 --fineSize 400 \
    --model multi_cycle_gan_semantic --no_flip --batchSize 4 \
    --dataset_mode gta_synthia_cityscapes --dataroot /nfs/project/libo_i/MADAN/data \
    --DSC --general_semantic_weight 20 --CCD --SAD --CCD_weight 0.2 --SAD_frozen_epoch 5 --CCD_frozen_epoch 10 --max_epoch 40 --gpu 0,1,2,3 \
    --weights_syn /nfs/project/libo_i/cycada/pretrained_models/cyclesynthia_V4_SEM_Final_iter_6000.pth \
    --weights_gta /nfs/project/libo_i/cycada/pretrained_models/drn26_cycada_cyclegta2cityscapes.pth \
    --gpu 0,1,2,3 --semantic_loss