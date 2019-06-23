#!/usr/bin/env bash
cd /root/MADAN/cyclegan

python3 train.py --name cycada_gta2cityscapes_A2B_noIdentity_SEM_KL \
    --resize_or_crop scale_width_and_crop --loadSize=600 --fineSize=500 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan_semantic_fcn --no_flip --batchSize 8 --nThreads 8 \
    --dataset_mode gta5_cityscapes --dataroot /nfs/project/libo_i/cycada/data \
    --fcn_model drn26 --weights_init /nfs/project/libo_i/cycada/pretrained_models/drn26_cycada_cyclegta2cityscapes.pth \
    --semantic_loss --DSC --DSC_weight 50 \
    --gpu 0,1,2,3