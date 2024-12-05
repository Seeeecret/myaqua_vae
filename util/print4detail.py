import torch
import os

# 数据集路径
output_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank64_extracted_lora_weights/normalized_data"

# 列出所有保存的 .pth 文件
files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(".pth")]

# 检查是否有文件存在
if not files:
    print("No .pth files found in the specified directory.")
else:
    # 随机读取一个文件
    file_to_read = files[0]  # 这里选择第一个文件，也可以使用 random.choice(files)
    print(f"Reading file: {file_to_read}")

    # 加载数据
    data = torch.load(file_to_read)

    # 打印文件内容的详细信息
    print("\nFile content summary:")
    for key, value in data.items():
        if key == "data":
            print(f"Key: {key}, Flattened data length: {value.shape}")
        else:
            print(f"Key: {key}")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")

    # 打印展平的参数数据的一部分（前10个值）
    if "data" in data:
        print("\nFlattened data (first 10 values):")
        print(data["data"][:10])
