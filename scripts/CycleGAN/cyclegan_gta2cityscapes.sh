#!/usr/bin/env bash
cd /nfs/project/libo_i/MADAN/cyclegan

sudo python3 train.py --name cyclegan_gta2cityscapes \
    --resize_or_crop scale_width_and_crop --loadSize 600 --fineSize 500 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan_semantic_fcn --no_flip --batchSize 2 --nThreads 8 \
    --dataset_mode gta5_cityscapes --dataroot /nfs/project/libo_i/MADAN/data \
    --model_type drn26 --weights_init /nfs/project/libo_i/MADAN/pretrained_models/drn26_cycada_cyclegta2cityscapes.pth \
    --semantic_loss --gpu 0