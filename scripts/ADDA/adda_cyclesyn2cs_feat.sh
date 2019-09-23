#!/usr/bin/env bash

gpu=0,1,2,3

######################
# loss weight params #
######################
lr=1e-5
momentum=0.99
lambda_d=1
lambda_g=0.1

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH='/usr/bin/python3'

################
# train params #
################
max_iter=100000
crop=800
snapshot=5000
batch=4

weight_share='weights_shared'
discrim='discrim_score'

########
# Data #
########
src='cyclesynthia'
tgt='cityscapes'
data_flag='V2_SEM'
datadir='/nfs/project/libo_i/cycada/data/'


resdir="results/${src}_to_${tgt}/adda_sgd/${weight_share}_nolsgan_${discrim}"

# init with pre-trained cyclegta5 model
#model='drn26'
#baseiter=115000
model='fcn8s'
baseiter=100000


base_model="/nfs/project/libo_i/cycada/pretrained_models/cyclesynthia_V2_SEM_fcn8s-iter21000.pth"
outdir="${resdir}/${model}/lr${lr}_crop${crop}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"
echo $outdir
echo $base_model

cd /nfs/project/libo_i/cycada

# Run python script #
python3 scripts/train_fcn_adda.py ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --lr ${lr} --momentum ${momentum} --gpu ${gpu} \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --weights_init ${base_model} --model ${model} \
    --"${weight_share}" --${discrim} --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot ${snapshot}
