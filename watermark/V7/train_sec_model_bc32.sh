#!/bin/bash

# 设置环境变量
# export VAR_NAME=value

# 启动训练命令
CUDA_VISIBLE_DEVICES=1,6,7 accelerate launch train_sec_decoder.py \
    --image_dir /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/evaluation/rank4_bits4_output_0323_0101/ \
    --label_path /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/evaluation/rank4_bits4_output_0323_0101/image_labels.json \
    --output_dir /baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/V7_checkpoints/bc32_0324_0101 \
    --batch_size 32 \
    --num_epochs 150 \
    --learning_rate 1e-4