#!/bin/bash

# 起始step
start=18601
# 结束step
end=18750

for ((i=start; i<=end; i++))
do
    # 构建源文件路径
    src="/mnt/share_disk/dorin/AquaLoRA/train/rank8_bits8_output/checkpoint-${i}/pytorch_lora_weights.safetensors"
    # 构建目标文件路径
    dest="/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank8_8bits_extracted_lora_weights/pytorch_lora_weights_${i}.safetensors"
    # 执行复制命令
    cp "$src" "$dest"
done
