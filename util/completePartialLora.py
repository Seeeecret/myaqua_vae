"""
    将残缺的Lora模型权重恢复为完整的模型权重
"""
import os
import torch
from safetensors.torch import load_file, save_file

# TODO:原始完整模型权重路径（未经过VAE采样等处理）
original_model_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank8_8bits_extracted_lora_weights/pytorch_lora_weights_18750.safetensors"
# 部分恢复的模型权重路径（只包含部分已逆归一化后的参数）
partial_model_path = "/mnt/share_disk/dorin/AquaLoRA/output/rank8_8bits_partial_lora_vae_checkpoints_1219_8000epoch/pytorch_lora_weights.safetensors"
# 输出完整模型权重存放目录
output_dir = "/mnt/share_disk/dorin/AquaLoRA/output/rank8_8bits_partial_lora_vae_checkpoints_1219_8000epoch/completed"
# output_dir = "/mnt/share_disk/dorin/AquaLoRA/output/rank8_8bits_partial_lora_vae_checkpoints_1219_150epoch/completed"
os.makedirs(output_dir, exist_ok=True)

# 输出文件名
output_file = os.path.join(output_dir, "pytorch_lora_weights.safetensors")

# 加载原始完整的state_dict
original_state_dict = load_file(original_model_path, device='cpu')

# 加载部分恢复的state_dict
partial_state_dict = load_file(partial_model_path, device='cpu')

# 将partial_state_dict中存在的参数更新到original_state_dict中
for key, value in partial_state_dict.items():
    if key in original_state_dict:
        # 使用partial中的参数覆盖original中的参数
        original_state_dict[key] = value
    else:
        pass

# 将合并后的完整模型权重保存
save_file(original_state_dict, output_file)

print(f"Complete model weights saved at: {output_file}")
