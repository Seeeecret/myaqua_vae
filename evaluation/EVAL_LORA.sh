#!/bin/bash

python run_eval_base.py --lora '/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/train/Ortho_rank8_bits8_output_0329_local/10101110/pytorch_lora_weights.safetensors' \
--output_dir /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/evaluation/output_Ora10101110  \
--msgdecoder /baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/train/Ortho_rank8_bits8_output_0329_local/msgdecoder.pt \
--msg_gt "10101110" \
--model "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5" \
--msg_bits 8