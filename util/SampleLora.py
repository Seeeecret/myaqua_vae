# sample_and_save.py

import torch
import os
from safetensors.torch import load_file
import sys
sys.path.append("..")

from myVAEdesign3_rank64 import OneDimVAE as VAE  # 导入您的 VAE 模型
# from myVAEdesign3_rank4 import OneDimVAE as VAE  # 导入您的 VAE 模型

def main():
    # ==============================
    # 1. 设置模型参数
    # ==============================
    latent_dim = 4096            # 与训练时相同
    input_length = 27131904  # 与训练数据长度相同
    # input_length = 1695744
    kld_weight = 0.005          # 与训练时相同

    # ==============================
    # 2. 创建模型实例并加载权重
    # ==============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型实例
    model = VAE(
        latent_dim=latent_dim,
        input_length=input_length,
        kld_weight=kld_weight
    ).to(device)

    # 加载模型权重
    checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank_64_lora_vae_checkpoints_1204/checkpoint_End/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank_64_lora_vae_checkpoints_1202/checkpoint_epoch_400/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/rank_64_lora_vae_checkpoints_1127/checkpoint_epoch_400/model.safetensors'  # 根据实际路径修改
    # checkpoint_path = '/mnt/share_disk/dorin/AquaLoRA/checkpoints/lora_vae_checkpoints_alter3/checkpoint_epoch_280/model.safetensors'  # 根据实际路径修改
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # 加载权重到模型中
    model.load_state_dict(load_file(checkpoint_path))
    print("Model weights loaded successfully.")

    # 设置模型为评估模式
    model.eval()

    # ==============================
    # 3. 采样生成数据
    # ==============================
    num_samples = 3  # 生成样本的数量
    with torch.no_grad():
        # 从标准正态分布中采样潜在向量 z
        generated_data = model.sample(num_samples).to(device)

    # ==============================
    # 4. 保存生成的数据
    # ==============================
    output_dir = '../generated_samples/rank64_kld_weight_0005/20241204'  # 保存生成数据的目录
    # output_dir = '../generated_samples/rank64_kld_weight_0005/20241202'  # 保存生成数据的目录
    # output_dir = '../generated_samples/rank4_kld_weight_0005/20241122'  # 保存生成数据的目录
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        sample = generated_data[i].cpu()
        sample_path = os.path.join(output_dir, f'sample_{i + 1}.pth')
        torch.save(sample, sample_path)
        print(f"Sample {i + 1} saved to {sample_path}")

    print("All samples generated and saved successfully.")

if __name__ == '__main__':
    main()
