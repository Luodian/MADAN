#!/usr/bin/env bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
cd /nfs/project/libo_i/MADAN

ckpt_path=$1
datadir=/nfs/project/libo_i/MADAN/data/cityscapes
model=fcn8s
num_cls=19
gpu=0

sudo python3 scripts/eval_fcn.py ${ckpt_path} \
        --dataset cityscapes \
        --datadir ${datadir} \
        --model ${model} --num_cls ${num_cls} \
        --gpu ${gpu}