import os
import time
import argparse
import torch
import random
from safetensors.torch import load_file, save_file


def main():
    # ----------------------
    # 1. 使用 argparse 解析命令行参数
    # ----------------------
    parser = argparse.ArgumentParser(description="Script for restoring LoRA state dict.")
    # 这里设置 3 个示例参数：
    # (1) `--reconstructed_lora_path` : 指定要加载的 .pth 文件
    # (2) `--format_data_path` : 指定与之对应的 “data 字典” .pth 文件
    # (3) `--output_dir` : 指定输出目录
    parser.add_argument(
        "--reconstructed_lora_path",
        type=str,
        required=True,
        help="Path to the .pth file for reconstructed lora vector."
    )
    parser.add_argument(
        "--format_data_path",
        type=str,
        required=True,
        help="Path to the .pth file with the normalized parameter info."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the restored state dict and safetensors."
    )
    args = parser.parse_args()

    # ----------------------
    # 2. 加载 reconstructed_lora_vector
    # ----------------------
    # 在原脚本中，此处为：
    # reconstructed_lora_vector = torch.load(rank8_8bits_kld_weight_0005_150_path1_1225)
    # TODO：选择采样的数据路径 -> 用命令行参数替代
    reconstructed_lora_vector = torch.load(args.reconstructed_lora_path)

    # 打印重建模型参数信息
    reconstructed_lora_param_info = {}

    if isinstance(reconstructed_lora_vector, dict):
        for key, value in reconstructed_lora_vector.items():
            print(key, value.shape)
            reconstructed_lora_param_info[key] = {
                'shape': value.shape,
                'length': value.numel()
            }
    else:
        print(reconstructed_lora_vector)
        print("Loaded data is not a dictionary. It might be a single Tensor.")

    # 打印观察前 1000 个数据
    print(reconstructed_lora_vector[:1000])

    # ----------------------
    # 3. 加载标准数据路径 (format_data_path)
    # ----------------------
    # 在原脚本中，此处为：
    # format_data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank8_8bits_extracted_lora_weights/normalized_data/normalized_pytorch_lora_weights_18750.pth"
    # TODO : 设置标准数据路径 -> 用命令行参数替代
    data_dict = torch.load(args.format_data_path)

    # 准备分割和还原
    data_keys = [k for k in data_dict.keys() if k != 'data']
    flattened_data = reconstructed_lora_vector
    lengths = [data_dict[k]['length'] for k in data_keys]

    print(f"Input tensor shape: {flattened_data.shape}")
    total_length = sum(lengths)
    print(f"Total length from split_sizes: {total_length}")
    print(f"Flattened data size: {flattened_data.shape[0]}")

    # 如果 flattened_data 不是 1D 张量，则压缩多余维度
    if len(flattened_data.shape) > 1:
        flattened_data = flattened_data.squeeze()
        print(f"Flattened data squeezed to shape: {flattened_data.shape}")

    split_data = torch.split(flattened_data, lengths)
    restored_state_dict = {}

    # ----------------------
    # 4. 还原参数
    # ----------------------
    for i, key in enumerate(data_keys):
        data_chunk = split_data[i]
        mean = data_dict[key]['mean']
        std = data_dict[key]['std']
        denormalized_data = data_chunk * std + mean
        # 重建成原始形状
        restored_state_dict[key] = denormalized_data.reshape(data_dict[key]['shape'])

    # 获取当前系统时间 (仅供打印或做其他用途)
    localtime = time.asctime(time.localtime(time.time()))

    # ----------------------
    # 5. 保存输出文件
    # ----------------------
    # 在原脚本中，此处为：
    # output_dir = '../output/rank8_8bits_kld000005_0102_800epoch'
    # TODO: 修改为自己的输出路径 -> 用命令行参数替代
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存成 .pth
    torch_output_dir = os.path.join(output_dir, 'restored_state_dict.pth')
    torch.save(restored_state_dict, torch_output_dir)

    # 保存成 .safetensors
    safetensors_output_dir = os.path.join(output_dir, 'pytorch_lora_weights.safetensors')
    save_file(restored_state_dict, safetensors_output_dir)

    print(f"Restored parameters have been saved to '{torch_output_dir}'")
    print(f"Restored safetensors saved to '{safetensors_output_dir}'")
    print(restored_state_dict)


if __name__ == "__main__":
    main()
