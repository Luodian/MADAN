#!/usr/bin/env bash
gpu=0,1,2,3
data=cyclesynthia
model=fcn8s

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

datadir=/root/MADAN/data
batch=28
iterations=20000
snapshot=1000
num_cls=19
data_flag=V4_SEM_Final_Scale

cd /root/MADAN

outdir=/root/MADAN/cycada/results/${data}/${data}_${data_flag}/${model}
mkdir -p results/${data}/${data}_${data_flag}/${model}
echo $outdir

python3 scripts/train_fcn.py ${outdir} --model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    -b ${batch} --adam \
    --iterations ${iterations} \
    --datadir ${datadir} \
    --snapshot ${snapshot} --small 1 \
    --dataset ${data} --data_flag ${data_flag}