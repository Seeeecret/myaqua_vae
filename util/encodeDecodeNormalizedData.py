import torch
import sys
from safetensors.torch import load_file

sys.path.append("..")

from myVAEdesign3_rank64 import OneDimVAE  # 假设上述模型代码保存在 my_vae_model.py 文件中

# 设置路径和设备
model_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank_64_lora_vae_checkpoints_1204/checkpoint_End/model.safetensors"
data_path = "/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_weights_dataset/rank64_extracted_lora_weights/normalized_data/normalized_pytorch_lora_weights_37499.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
latent_dim = 4096  # 与训练时保持一致
input_length = 27131904  # 输入数据的长度
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

output_path = "/mnt/share_disk/dorin/AquaLoRA/output/rank64_alter3_kld_weight_0005_1204/reconstructed_pytorch_lora_weights_37499/reconstructed_data.pth"
# 可选：保存重建结果
torch.save(reconstructed_data, output_path)
