#!/bin/bash

# 设置环境变量
# export VAR_NAME=value

# 启动训练命令
accelerate launch TrainScript5_V3_1.py     \
    --train_data_dir /baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/lora/normalized_data \
    --checkpoint_dir /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/checkpoints/WMVAE_V3_alter_rank4_4bits_3000epoch_1000iters_valc/  \
    --num_epochs 3000 \
    --save_checkpoint_epochs 2400  \
    --input_length 1695744 \
    --output_dir /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/output/WMVAE_V3_alter_rank4_4bits_3000epoch_1000iters_valc\
    --batch_size 20 \
     --log_dir /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/logs/WMVAE_V3_alter_rank4_4bits_3000epoch_1000iters_valc.log \
     --n_iters 1000
