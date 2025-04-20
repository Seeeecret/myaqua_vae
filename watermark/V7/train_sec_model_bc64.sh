#!/bin/bash

# 设置环境变量
# export VAR_NAME=value

# 启动训练命令
accelerate launch train_sec_decoder.py \
    --image_dir /baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/V7_img_dataset \
    --label_path /baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/V7_img_dataset/merged_labels.json \
    --output_dir /baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/V7_checkpoints/bc64_0324_200epoch \
    --batch_size 64 \
    --num_epochs 200 \
    --learning_rate 1e-4
