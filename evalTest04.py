# sample_and_save.py

import torch
import os
from myVAEdesign3 import VAE  # 导入您的 VAE 模型

def main():
    # ==============================
    # 1. 设置模型参数
    # ==============================
    latent_dim = 512          # 与训练时相同
    input_length = 135659520  # 与训练数据长度相同
    kld_weight = 0.005          # 与训练时相同
    encoder_channel_list = [1, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 256]
    decoder_channel_list = [256, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2, 1]

    # ==============================
    # 2. 创建模型实例并加载权重
    # ==============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型实例
    model = VAE(
        latent_dim=latent_dim,
        input_length=input_length,
        kld_weight=kld_weight,
        encoder_channel_list=encoder_channel_list,
        decoder_channel_list=decoder_channel_list
    ).to(device)

    # 加载模型权重
    checkpoint_path = './output/sum_kld_weight_0005/vae_final.pth'  # 请根据实际路径修改
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    # 加载权重到模型中
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    output_dir = './generated_samples/sum_kld_weight_0005/20241104'  # 保存生成数据的目录
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        sample = generated_data[i].cpu()
        sample_path = os.path.join(output_dir, f'sample_{i + 1}.pth')
        torch.save(sample, sample_path)
        print(f"Sample {i + 1} saved to {sample_path}")

    print("All samples generated and saved successfully.")

if __name__ == '__main__':
    main()
