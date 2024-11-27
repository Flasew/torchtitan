#!/bin/bash

jobs=( llama2-7be llama2-13be llama3-8be )
#jobs=( llama2-7b llama2-13b llama3-8b )

eval "$(conda shell.bash hook)"
conda activate ckpt

sleep 3

source /data/frankwwy/innet_ckpt/script/set_dpdk_env.sh

for job in ${jobs[@]}; do
   NCCL_BUFFSIZE=1048576 NCCL_P2P_NET_CHUNKSIZE=32768 sudo -E nice -n -20 numactl -m 0 -C 0-7,24-31 /home/frankwwy/miniconda3/envs/ckpt/bin/python /home/frankwwy/miniconda3/envs/ckpt/bin/torchrun --nnodes=8 --nproc-per-node=1 --node-rank=$MY8RANK --rdzv-id=1 --rdzv-backend=c10d --rdzv-endpoint=midna train.py --job.config_file ./train_configs/${job}.toml --training.steps 20 # --job.save_to_file $job-dpdk
  sleep 1
done
