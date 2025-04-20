#!/bin/bash

# 设置环境变量
export TORCH_HOME=/baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/torch_cache

# 启动训练命令
python latent_wm_pretrain_alter.py \
--output_dir /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/train/output_bitnum32_40epoch_0329 \
--epochs 40 \
--dataset "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/coco/coco2017/train2017" \
--pretrained_model_name_or_path "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5" \
--bit_num 48
