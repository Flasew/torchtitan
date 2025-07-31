#!/bin/bash

#jobs=( gpt2 gpt3xl gpt3-7b )
jobs=( gpt3-7b )

eval "$(conda shell.bash hook)"
conda activate ckpt

sleep 3

source /data/frankwwy/innet_ckpt/script/set_ib_env.sh

for job in ${jobs[@]}; do
  /home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=12 --nproc-per-node=1 --node-rank=$MY12RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=uther train.py --job.config_file ./train_configs/${job}.toml --checkpoint.checkfreq --checkpoint.enable_checkpoint --checkpoint.folder /sinkhole --job.save_to_file $job-ib
  sleep 10
done
