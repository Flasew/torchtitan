#!/bin/bash

#jobs=( llama2-7b llama3-8b llama2-13b )
jobs=( llama2-7b llama3-8b )

eval "$(conda shell.bash hook)"
conda activate ckpt

sleep 3

source /data/frankwwy/innet_ckpt/script/set_ib_env.sh

for job in ${jobs[@]}; do
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=12 --nproc-per-node=1 --node-rank=$MY12RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=uther train.py --job.config_file ./train_configs/${job}.toml --checkpoint.checkfreq --checkpoint.enable_checkpoint --checkpoint.folder /sinkhole --training.steps 1000 --job.save_to_file $job-ib
  sleep 10
done
