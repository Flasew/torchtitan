#!/bin/bash

#jobs=( gpt2 gpt3xl gpt3-7b )
jobs=( gpt2 gpt3xl )
FREQ=1

eval "$(conda shell.bash hook)"

sleep 3

source /data/frankwwy/innet_ckpt/script/set_ib_env.sh

for job in ${jobs[@]}; do
  /home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=12 --nproc-per-node=1 --node-rank=$MY12RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=uther train.py --job.config_file ./train_configs/${job}.toml --checkpoint.enable_checkpoint --checkpoint.folder /sinkhole --checkpoint.interval $FREQ --checkpoint.export_dtype bfloat16 --checkpoint.async_mode async_with_pinned_mem --job.save_to_file $job-ib
  sleep 10
done
