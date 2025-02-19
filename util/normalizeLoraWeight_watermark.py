import glob
import torch
from safetensors.torch import load_file
import os

# 数据集路径，请根据实际情况修改
dataset_path = "/data/Tsinghua/wuzy/rank4_bits4_output_0203"
output_path = os.path.join(dataset_path, "normalized_partial_data_wm")
output_path_wwm = os.path.join(dataset_path, "normalized_partial_data_wwm")

os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path_wwm, exist_ok=True)

for file in glob.glob(os.path.join(dataset_path, "*.safetensors")):
    model = load_file(file, device='cpu')
    single_lora_dict = {}
    single_lora_weights = []

    # 用于保存不符合条件的参数
    single_lora_dict_wwm = {}
    single_lora_weights_wwm = []

    for key, value in model.items():
        # 仅对特定条件的参数进行处理
        # 条件：
        # 1. key中包含'unet'
        # 2. 对'attn' 或 'ff'层中包含'down.weight'的参数处理
        # 3. 对'proj_in' 或 'proj_out'层中包含'down.weight'的参数处理

        if 'unet' in key:
            if (('attn' in key or 'ff' in key) and 'down.weight' in key) or \
               (('proj_in' in key or 'proj_out' in key) and 'down.weight' in key):
                # 对符合条件的权重进行标准化处理
                mean_val = value.mean()
                std_val = value.std()
                # 避免除以0的情况
                if std_val == 0:
                    std_val = 1e-11
                normalized = (value - mean_val) / std_val

                # 将处理结果保存
                flattened_normalized = normalized.flatten()
                single_lora_weights.append(flattened_normalized)
                single_lora_dict[key] = {
                    "mean": mean_val.item(),
                    "std": std_val.item(),
                    "length": value.numel(),
                    "shape": value.shape,
                }
            else:
                # 不符合条件的权重则在同样的处理后保存到wwm路径
                mean_val = value.mean()
                std_val = value.std()
                if std_val == 0:
                    std_val = 1e-11
                normalized = (value - mean_val) / std_val

                flattened_normalized = normalized.flatten()
                single_lora_weights_wwm.append(flattened_normalized)
                single_lora_dict_wwm[key] = {
                    "mean": mean_val.item(),
                    "std": std_val.item(),
                    "length": value.numel(),
                    "shape": value.shape,
                }



    # 将所有处理过的参数展平并保存
    if len(single_lora_weights) > 0:
        single_lora_weights = torch.cat(single_lora_weights, dim=0)
        single_lora_dict["data"] = single_lora_weights

        # 输出信息
        print(f"File: {file}, Flattened length: {single_lora_weights.shape}")

        # 保存处理结果
        torch.save(
            single_lora_dict,
            os.path.join(output_path, f"normalized_{os.path.basename(file).split('.')[0]}.pth")
        )
    else:
        print(f"File: {file} 没有符合条件的权重需要处理。")

    # 将所有处理过的wwm参数展平并保存
    if len(single_lora_weights_wwm) > 0:
        single_lora_weights_wwm = torch.cat(single_lora_weights_wwm, dim=0)
        single_lora_dict_wwm["data"] = single_lora_weights_wwm

        # 输出信息
        print(f"File: {file}, Flattened length: {single_lora_weights_wwm.shape}")

        # 保存处理结果
        torch.save(
            single_lora_dict_wwm,
            os.path.join(output_path_wwm, f"normalized_{os.path.basename(file).split('.')[0]}.pth")
        )
    else:
        print(f"File: {file} 没有符合条件的权重需要处理。")
