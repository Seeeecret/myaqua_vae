import glob
import os
import torch

# 假设 normalized_partial_data 是生成脚本中保存 .pth 文件的目录
dataset_path = "/data/Tsinghua/wuzy/rank8_bits8_dataset/normalized_partial_data_9360"

# 遍历目录下所有的 pth 文件
for pth_file in glob.glob(os.path.join(dataset_path, "*.pth")):
    # 1. 加载 pth 文件
    single_lora_dict = torch.load(pth_file, map_location="cpu")

    # 2. 读取并打印 "data" 的基本信息
    #    这个 "data" 对应脚本中 single_lora_dict["data"] = single_lora_weights
    data = single_lora_dict.get("data", None)
    if data is None:
        print(f"File: {pth_file} 不包含 'data' 字段，跳过。")
        continue

    # data 是所有满足条件的参数拼接后的扁平化张量
    print(f"文件: {pth_file}")
    print(f"  -> data 大小: {data.shape}, 总元素数: {data.numel()}")

    # 3. 遍历并解析除 "data" 以外的键，提取各自的元信息，并从 data 中切片还原
    index = 0
    for key, info_dict in single_lora_dict.items():
        # 跳过 "data" 本身
        if key == "data":
            continue

        # info_dict 里包含了 mean, std, length, shape 等信息
        length = info_dict["length"]
        shape = info_dict["shape"]

        # 根据 length 和 shape 从 data 中切片还原该参数对应的张量
        param_slice = data[index: index + length].view(shape)
        index += length

        # 简单分析：打印还原后的张量的 mean/std，与保存时记录的 mean/std 比较
        current_mean = param_slice.mean().item()
        current_std = param_slice.std().item()

        print(f"  Key: {key}")
        print(f"    - 记录形状: {shape}")
        print(f"    - 读取后 mean: {current_mean:.6f}, std: {current_std:.6f}")
        print(f"    - 生成时 mean: {info_dict['mean']:.6f}, std: {info_dict['std']:.6f}")

    print("-" * 60)

print("所有 pth 文件读取与分析完毕。")
