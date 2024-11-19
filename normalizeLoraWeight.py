import glob

import torch
from safetensors.torch import load_file
import os

dataset_path = "/mnt/share_disk/dorin/AquaLoRA/train/output_from_beginning/steploras"
output_path = os.path.join(dataset_path, "normalized_data")
os.makedirs(output_path, exist_ok=True)

# 用于计算整体 mean 和 std 的容器
all_parameters = []

# 遍历每个 safetensors 文件
for file in glob.glob(os.path.join(dataset_path, "*.safetensors")):
    single_lora_dict = {}  # 用于保存单个文件的所有信息
    single_lora_weights = []  # 保存展平的参数

    model = load_file(file)
    for key, value in model.items():
        flattened_value = value.flatten()  # 将每个参数展平为一维
        single_lora_weights.append(flattened_value)
        all_parameters.append(flattened_value)  # 加入全局统计

        # 保存当前参数的统计信息到字典
        single_lora_dict[key] = {
            "mean": value.mean().item(),
            "std": value.std().item(),
            "normalized": (value - value.mean()) / value.std()
        }

    # 将该文件的参数展平并保存到字典
    single_lora_weights = torch.cat(single_lora_weights, dim=0)
    single_lora_dict["data"] = single_lora_weights

    # 打印该文件的参数展平后的长度
    print(f"File: {file}, Flattened length: {single_lora_weights.shape}")

    # 保存该文件的字典
    torch.save(
        single_lora_dict,
        os.path.join(output_path, f"normalized_{os.path.basename(file).split('.')[0]}.pth")
    )