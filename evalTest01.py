# sample_and_save.py

import torch
import os
from myVAEdesign3 import VAE  # 导入您的 VAE 模型

def main():
    # ==============================
    # 1. 设置模型参数
    # ==============================
    latent_dim = 256          # 与训练时相同
    input_length = 135659520  # 与训练数据长度相同
    kld_weight = 0.5          # 与训练时相同

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
    checkpoint_path = './output/kld_weight_05/vae_final.pth'  # 请根据实际路径修改
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

        # 如果需要对生成的数据进行后处理，可以在这里添加代码
        # 例如，将数据范围从 [-1, 1] 转换为原始数据的范围

    # ==============================
    # 4. 保存生成的数据
    # ==============================
    output_dir = './generated_samples/kld_weight_05/20241101'  # 保存生成数据的目录
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        sample = generated_data[i].cpu()  # 将数据移动到 CPU
        sample_path = os.path.join(output_dir, f'sample_{i + 1}.pth')
        torch.save(sample, sample_path)
        print(f"Sample {i + 1} saved to {sample_path}")

    print("All samples generated and saved successfully.")

if __name__ == '__main__':
    main()
