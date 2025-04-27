#!/bin/bash

# 设置环境变量
# export VAR_NAME=value

# 启动训练命令
python latent_wm_pretrain_alter.py \
--output_dir /baai-cwm-1/baai_cwm_ml/cwm/ziyang.yan/myaqua_vae/train/output_bitnum64_40epoch_valc \
--epochs 40 \
--dataset "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/coco/coco2017/train2017" \
--pretrained_model_name_or_path "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5" \
--bit_num 64
