#!/usr/bin/env bash
cd /root/MADAN/cyclegan

python3 train.py --name cycada_synthia2cityscapes_A2B_noIdentity_sem_V4_KL100_final_Scale \
    --resize_or_crop scale_width_and_crop --loadSize=500 --fineSize=400 --model cycle_gan_semantic_fcn --no_flip --batchSize 12 --nThreads 8 \
    --dataset_mode synthia_cityscapes --dataroot /nfs/project/libo_i/cycada/data \
    --fcn_model fcn8s --weights_init /nfs/project/libo_i/cycada/pretrained_models/cyclesynthia_V4_SEM_Final_fcn8s_iter_7000.pth \
    --semantic_loss --DSC --DSC_weight 20 \
    --gpu 0,1,2,3