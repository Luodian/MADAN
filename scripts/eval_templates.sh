#!/usr/bin/env bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
cd /nfs/project/libo_i/cycada

python3 scripts/eval_fcn.py $1 \
        --dataset cityscapes \
        --datadir /nfs/project/libo_i/cycada/data/cityscapes \
        --model $2 --num_cls $3 --small 1 \
        --gpu 0,1,2,3