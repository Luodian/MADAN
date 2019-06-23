#!/usr/bin/env bash

gpu=0,1,2,3

######################
# loss weight params #
######################
lr=2e-5
momentum=0.9
lambda_d=1
lambda_g=0.1

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH='/usr/bin/python3'

################
# train params #
################
max_iter=50000
crop=600
snapshot=5000
batch=8

weight_share='weights_shared'
discrim='discrim_feat'

########
# Data #
########
src='cyclegta5'
tgt='cityscapes'
data_flag='V4_SEM_300'
datadir='/nfs/project/libo_i/cycada/data/'


resdir="results/${src}_to_${tgt}/adda_sgd_${weight_share}_nolsgan_${discrim}_${data_flag}"

# init with pre-trained cyclegta5 model
#model='drn26'
#baseiter=115000
model='fcn8s'
baseiter=100000

base_model="/nfs/project/libo_i/cycada/pretrained_models/GTA2CS_score_V4_SEM_300_net-itercurr.pth"
discrim_model="/nfs/project/libo_i/cycada/pretrained_models/new_Dis_cyclegta5_sem_v4_300x540_iter-iter5440_abv60.pth"
outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}_${discrim}"

echo $outdir
echo $base_model

cd /nfs/project/libo_i/cycada

# Run python script #
python3 scripts/train_fcn_adda.py \
    ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --lr ${lr} --momentum ${momentum} --gpu ${gpu} \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --weights_init ${base_model} --model ${model} \
    --"${weight_share}" --${discrim} --no_lsgan \
    --max_iter ${max_iter} --batch ${batch} \
    --snapshot ${snapshot} --no_mmd_loss --small 0 --resize 300 --data_flag ${data_flag}
