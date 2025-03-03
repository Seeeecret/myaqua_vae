#!/bin/bash

# 设置环境变量
# export VAR_NAME=value

# 启动训练命令
accelerate launch TrainScript5_V3_1.py     \
    --train_data_dir /baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/juliensimon/stable-diffusion-v1-5-pokemon-lora/normalized_data \
    --checkpoint_dir /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/checkpoints/WMVAE_pokemon_V3_rank4_3000epoch_2000iters_valc/  \
    --num_epochs 3000 \
    --save_checkpoint_epochs 2400  \
    --input_length 797184 \
    --output_dir /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/output/WMVAE_pokemon_V3_rank4_3000epoch_2000iters_valc \
    --batch_size 10 \
     --log_dir /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/logs/WMVAE_pokemon_V3_rank4_3000epoch_2000iters_valc.log \
     --n_iters 2000

