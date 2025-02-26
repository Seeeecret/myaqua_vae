#!/bin/bash

# 设置环境变量
# export VAR_NAME=value

# 启动训练命令
accelerate launch --main_process_port 29554 TrainScript5_V3.py     \
    --train_data_dir /baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/lora/normalized_data \
    --checkpoint_dir /baai-cwm-nas/algorithm/ziyang.yan/ckpt/WMVAE_small_V3_rank4_4bits_3000epoch/  \
    --num_epochs 3000 \
    --save_checkpoint_epochs 2400  \
    --input_length 1695744 \
    --output_dir /baai-cwm-nas/algorithm/ziyang.yan/output/WMVAE_small_V3_rank4_4bits_3000epoch \
    --batch_size 8 \
     --log_dir /baai-cwm-nas/algorithm/ziyang.yan/logs/WMVAE_small_V3_rank4_4bits_3000epoch.log
