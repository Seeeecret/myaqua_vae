import glob
import torch
from safetensors.torch import load_file
import os

# 数据集路径，请根据实际情况修改
dataset_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank8_8bits_extracted_lora_weights"
output_path = os.path.join(dataset_path, "normalized_partial_full_data")
os.makedirs(output_path, exist_ok=True)

for file in glob.glob(os.path.join(dataset_path, "*.safetensors")):
    model = load_file(file, device='cpu')
    single_lora_dict = {}
    single_lora_weights = []

    for key, value in model.items():
        # 对第二份代码中的判断条件进行还原：
        # if 'unet' in key:
        #     if ('attn' in key or 'ff' in key):
        #         if 'up.weight' in key or 'down.weight' in key:
        #             # 此处需要进行处理(类似代码一)
        #     if ('proj_in' in key or 'proj_out' in key):
        #         if 'up.weight' in key or 'down.weight' in key:
        #             # 此处需要进行处理(类似代码一)

        if 'unet' in key:
            # 判断是否为 attn/ff 层对应的 up/down.weight
            condition_attn_ff = ('attn' in key or 'ff' in key) and ('up.weight' in key or 'down.weight' in key)
            # 判断是否为 proj_in/proj_out 层对应的 up/down.weight
            condition_proj = ('proj_in' in key or 'proj_out' in key) and ('up.weight' in key or 'down.weight' in key)

            if condition_attn_ff or condition_proj:
                # 进行与第一份代码类似的标准化处理
                mean_val = value.mean()
                std_val = value.std()
                if std_val == 0:
                    std_val = 1e-10  # 避免除0

                normalized = (value - mean_val) / std_val
                flattened_normalized = normalized.flatten()

                single_lora_weights.append(flattened_normalized)
                single_lora_dict[key] = {
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
