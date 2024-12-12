import os

import torch
import sys
from safetensors.torch import load_file
import torch.nn.functional as F

sys.path.append("..")

# from myVAEdesign3_rank64 import OneDimVAE  # 假设上述模型代码保存在 my_vae_model.py 文件中
# from myVAEdesign3_rank4 import OneDimVAE  # 假设上述模型代码保存在 my_vae_model.py 文件中
from myVAEdesign3_SHAO import OneDimVAE  # 假设上述模型代码保存在 my_vae_model.py 文件中

# 设置路径和设备
# model_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank_64_lora_vae_checkpoints_1204/checkpoint_End/model.safetensors"
# model_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_vae_checkpoints_alter3_1206/checkpoint_End/model.safetensors"
# model_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/SHAO_lora_vae_checkpoints_1208/checkpoint_End/model.safetensors"
model_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/SHAO_lora_vae_checkpoints_1209/checkpoint_End/model.safetensors"

# data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank64_extracted_lora_weights/normalized_data/normalized_pytorch_lora_weights_37499.pth"
# data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank4_extracted_lora_weights/normalized_data/normalized_pytorch_lora_weights_37499.pth"
# data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/bus/normalized_data/normalized_adapter_model_31.pth"
data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/bus/flat_data/flat_adapter_model_31.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
# TODO:定义模型参数
latent_dim = 4096  # 与训练时保持一致
# input_length = 27131904  # 输入数据的长度
# input_length = 1695744  # 输入数据的长度
input_length = 1929928  # 输入数据的长度
vae = OneDimVAE(latent_dim=latent_dim, input_length=input_length, kld_weight=0.005).to(device)

# 加载模型权重
checkpoint = load_file(model_path)
vae.load_state_dict(checkpoint)
vae.eval()

# 准备测试数据
# 加载测试数据
test_data_dict = torch.load(data_path, map_location=device)
test_data = test_data_dict["data"]  # 提取展平的参数数据
test_data = test_data.unsqueeze(0).unsqueeze(0)  # 调整形状为 (1, 1, input_length)

# 重建数据
with torch.no_grad():
    reconstructed_data, input, _, _ = vae.encode_decode(test_data)


# 打印输入和输出数据的形状
print("Input Data Shape:", test_data.shape)
print("Reconstructed Data Shape:", reconstructed_data.shape)

# 打印后 100 个值对比
# print("Original Data (last 100 values):", test_data.flatten()[-100:].tolist())
# print("Reconstructed Data (last 100 values):", reconstructed_data.flatten()[-100:].tolist())

# 确保两者形状一致，裁剪或调整为相同的长度
if test_data.shape != reconstructed_data.shape:
    min_length = min(test_data.shape[-1], reconstructed_data.shape[-1])
    test_data = test_data[..., :min_length]
    reconstructed_data = reconstructed_data[..., :min_length]

# 打印前 100 个值对比
print("Original Data (first 100 values):", test_data.flatten()[:100])
print("Reconstructed Data (first 100 values):", reconstructed_data.flatten()[:100])


# 计算误差（例如均方误差）
# 先让测试数据对齐重建数据的形状

test_data = test_data.to(reconstructed_data.device)
test_data = test_data.reshape(reconstructed_data.shape)
# 使用F.mse_loss()计算均方误差
mse = F.mse_loss(reconstructed_data, test_data).item()
# mse = torch.mean((test_data - reconstructed_data) ** 2).item()
print(f"Mean Squared Error (MSE): {mse}")

# TODO: 修改输出路径
# output_path = "/mnt/share_disk/dorin/AquaLoRA/output/rank4_alter3_kld_weight_00005_1206/reconstructed_pytorch_lora_weights_37499/reconstructed_data.pth"
output_path = "/mnt/share_disk/dorin/AquaLoRA/output/SHAO_alter3_kld_weight_0005_1209/test_reconstructed_flat_adapter_model_31/reconstructed_data.pth"
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
# 保存重建结果
torch.save(reconstructed_data, output_path)

print(f"Reconstructed data saved to {output_path}")
