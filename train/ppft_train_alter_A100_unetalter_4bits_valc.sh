#!/bin/bash

accelerate launch --mixed_precision="fp16" ppft_train_alter_A100.py \
  --pretrained_model_name_or_path="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5" \
  --train_data_dir="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/train_512_512" --caption_column="text" \
  --resolution=512 \
  --dataloader_num_workers=12 \
  --train_batch_size=8 \
  --num_train_epochs=40 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=0 --lr_end=0.01 \
  --seed=2048 \
  --output_dir="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/train_ank4_bits4_output_0321_valc" \
  --start_from_pretrain="4bit_state_dict_29.pth" \
  --validation_prompt="A portrait of a young white woman, masterpiece" \
  --validation_epochs=1 \
  --rank=4 \
  --msg_bits=4