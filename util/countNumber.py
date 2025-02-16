import safetensors
import torch
from safetensors.torch import load_file


def count_params(model_file):
    # 读取 safetensors 格式的权重文件
    tensors = load_file(model_file)

    total_params = 0
    # 遍历所有的 tensor，计算参数总量
    for name, tensor in tensors.items():
        total_params += tensor.numel()  # numel() 返回 tensor 中所有元素的总数

    return total_params


# 示例: 计算 safetensors 格式文件的参数量
# model_file = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/rank8_8bits_lora_vae_checkpoints_0110_800epoch/checkpoint_End/model.safetensors'  # 替换为你的 safetensors 文件路径
model_file = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/SHAO_dog_r8_voc_4000epoch_0113/checkpoint_End/model.safetensors'  # 替换为你的 safetensors 文件路径
param_count = count_params(model_file)

print(f"Total number of parameters: {param_count}")

# file_path = '/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/checkpoints/rank8_8bits_lora_vae_checkpoints_0110_800epoch/checkpoint_End/model.safetensors'
# total_params = calculate_model_parameters(file_path)
# print(f"Total number of parameters: {total_params}")
